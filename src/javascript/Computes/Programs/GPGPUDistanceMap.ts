
import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import * as su from '../../Utils/ShadowMapUtils'
import { type Sign, type Octant, type Axis } from '../../Utils/ShadowMapUtils'

/**
 * Scalar Chebyshev distance transforms for shadow masks.
 *
 * Masks are treated as binary fields where 0-valued cells are distance seeds
 * and shadowed/rejected cells start at maxDistance. The passes then propagate
 * the nearest seed distance either isotropically, in one octant direction, or
 * in the extended anisotropic layout used by the MIP ray marcher.
 */

/**
 * Logs the mean over spatial axes without downloading the full tensor.
 */
function logMean3d(
    name: string, 
    tensor: tf.Tensor
): void
{
    console.log(name, tf.tidy(() => tensor.mean([0, 1, 2]).dataSync()))
}

/**
 * Runs a custom WebGL program and wraps the backend TensorInfo as a Tensor.
 */
function runWebGLProgram(
    program: GPGPUProgram,
    inputs: tf.Tensor[],
    dtype?: tf.DataType,
    uniforms?: number[][],
    preventEagerUnpackingOfOutput?: boolean
): tf.Tensor
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(program, inputs, dtype, uniforms, preventEagerUnpackingOfOutput)

    return tf.engine().makeTensorFromTensorInfo(info)
}

/**
 * Converts a binary shadowMap into initial distances: clear cells become zero-distance
 * seeds, while masked cells start at the caller's maximum representable range.
 */
class InitialChebyshevDistancePass implements GPGPUProgram 
{
    variableNames = ['A'] 
    outputShape: [number, number, number]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        shape: [number, number, number],
        maxDistance: number,
    ) {
        this.outputShape = shape
        this.userCode = `
        ivec3 outputCoords() 
        {
            ivec3 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        float cellShadowAt(ivec3 coords) 
        {
            return step(0.5, getA(coords.z, coords.y, coords.x));
        }

        void main() 
        {
            float cellShadow = cellShadowAt(outputCoords());
            float cellDistance = mix(float(0), float(${maxDistance}), cellShadow);
            
            setOutput(cellDistance);
        }
        `
    }
}

/**
 * Symmetric 1D Chebyshev transform along one axis. A candidate at radius r
 * contributes max(previousDistance, r), so chaining x/y/z produces a full
 * isotropic Chebyshev distance map.
 */
class IsotropicChebyshevDistancePass implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        shape: [number, number, number],
        sweepAxis: Axis,
        maxDistance: number, 
    ) { 
        const [depth, height, width,] = shape

        this.outputShape = shape
        this.userCode = /* glsl */ `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        const int maxRadius = clamp(${maxDistance}, minCoords.${sweepAxis}, maxCoords.${sweepAxis});

        bool insideAxis(ivec3 cellCoords)
        {
            return (cellCoords.${sweepAxis} >= minCoords.${sweepAxis} && cellCoords.${sweepAxis} <= maxCoords.${sweepAxis});
        }

        ivec3 outputCoords()
        {
            ivec3 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        int sampleChebyshevDistanceAt(ivec3 cellCoords)
        {
            return int(getA(cellCoords.z, cellCoords.y, cellCoords.x));
        }
    
        void main()
        {
            ivec3 outCoords = outputCoords();
            ivec3 sampleCoords = outCoords;

            int minDistance = sampleChebyshevDistanceAt(sampleCoords);

            if (minDistance == 0)
            {
                setOutput(float(minDistance));
                return;
            }

            for (int radius = 1; radius <= maxRadius; ++radius)
            {

                sampleCoords.${sweepAxis} = outCoords.${sweepAxis} - radius;
                
                if (insideAxis(sampleCoords))
                {
                    int sampleDistance = sampleChebyshevDistanceAt(sampleCoords);
                    int candidateDistance = max(sampleDistance, radius);

                    minDistance = min(minDistance, candidateDistance);

                    if (minDistance <= radius) 
                    {
                        break;
                    }
                }

                sampleCoords.${sweepAxis} = outCoords.${sweepAxis} + radius;

                if (insideAxis(sampleCoords))
                {
                    int sampleDistance = sampleChebyshevDistanceAt(sampleCoords);
                    int candidateDistance = max(sampleDistance, radius);

                    minDistance = min(minDistance, candidateDistance);

                    if (minDistance <= radius)
                    {
                        break;
                    }
                }
            }

            setOutput(float(minDistance));
        }
        `
    }
}

/**
 * One-sided 1D Chebyshev transform for an octant direction. This is the same
 * recurrence as the isotropic pass, but it only searches along the requested
 * sign of the sweep axis.
 */
class AnisotropicChebyshevDistancePass implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        shape: [number, number, number],
        sweepAxis: Axis,
        sweepSign: Sign,
        maxDistance: number,
    ) {
        const [depth, height, width,] = shape

        this.outputShape = shape
        this.userCode = /* glsl */ `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        const int maxRadius = clamp(${maxDistance}, minCoords.${sweepAxis}, maxCoords.${sweepAxis});

        bool outsideAxis(ivec3 cellCoords)
        {
            return (cellCoords.${sweepAxis} < minCoords.${sweepAxis} || cellCoords.${sweepAxis} > maxCoords.${sweepAxis});
        }

        ivec3 outputCoords()
        {
            ivec3 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        int sampleChebyshevDistanceAt(ivec3 cellCoords)
        {
            return int(getA(cellCoords.z, cellCoords.y, cellCoords.x));
        }

        void main()
        {
            ivec3 outCoords = outputCoords();
            ivec3 sampleCoords = outCoords;

            int minDistance = sampleChebyshevDistanceAt(sampleCoords);

            if (minDistance == 0)
            {
                setOutput(float(minDistance));
                return;
            }

            for (int radius = 1; radius <= maxRadius; ++radius)
            {
                sampleCoords.${sweepAxis} = outCoords.${sweepAxis} ${sweepSign} radius;

                if (outsideAxis(sampleCoords)) 
                {
                    break;
                }
                
                int sampleDistance = sampleChebyshevDistanceAt(sampleCoords);
                int candidateDistance = max(sampleDistance, radius);

                minDistance = min(minDistance, candidateDistance);

                if (minDistance <= radius) 
                {
                    break;
                }
            
            }

            setOutput(float(minDistance));
        }
        `
    }
}

/**
 * Final dominant-axis pass for the extended anisotropic distance. A sample can
 * support radius r only when the already-computed transverse distance reaches
 * that radius; otherwise the candidate remains maxDistance.
 */
class ExtendedChebyshevDistancePass implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: [number, number, number]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        shape: [number, number, number],
        dominantAxis: Axis,
        dominantSign: Sign,   
        maxDistance: number,     
    ) {
        const [depth, height, width,] = shape

        this.outputShape = shape
        this.userCode = /* glsl */ `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        const int maxRadius = clamp(${maxDistance}, minCoords.${dominantAxis}, maxCoords.${dominantAxis});

        bool outsideAxis(ivec3 cellCoords)
        {
            return (cellCoords.${dominantAxis} < minCoords.${dominantAxis} || cellCoords.${dominantAxis} > maxCoords.${dominantAxis});
        }

        ivec3 outputCoords()
        {
            ivec3 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        int sampleChebyshevDistanceAt(ivec3 cellCoords)
        {
            return int(getA(cellCoords.z, cellCoords.y, cellCoords.x));
        }

        int getCandidateDistance(int sampleDistance, int radius)
        {
            return sampleDistance <= radius ? radius : ${maxDistance};
        }

        void main() 
        {
            ivec3 outCoords = outputCoords();
            ivec3 sampleCoords = outCoords;

            int minDistance = ${maxDistance};

            for (int radius = 0; radius <= maxRadius; ++radius) 
            {
                sampleCoords.${dominantAxis} = outCoords.${dominantAxis} ${dominantSign} radius;

                if (outsideAxis(sampleCoords)) 
                {
                    break;
                }

                int sampleDistance = sampleChebyshevDistanceAt(sampleCoords);
                int candidateDistance = getCandidateDistance(sampleDistance, radius);
                
                minDistance = min(minDistance, candidateDistance); 
                
                if (minDistance <= radius) 
                {
                    break;
                }
                
            }

            setOutput(float(minDistance));
        }
        `
    }
}

function initialChebyshevDistancePass(
    cellShadows: tf.Tensor3D,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = cellShadows.shape as [number, number, number]
    const program = new InitialChebyshevDistancePass(shape, maxDistance)
    const initialDistances = runWebGLProgram(program, [cellShadows], 'int32', [], true) 
    if (verbose) logMean3d('initialDistances', initialDistances)

    return initialDistances as tf.Tensor3D
}

function isotropicChebyshevDistancePass(
    distances: tf.Tensor3D,
    sweepAxis: Axis,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = distances.shape as [number, number, number]
    const program = new IsotropicChebyshevDistancePass(shape, sweepAxis, maxDistance)
    const isotropicDistances = runWebGLProgram(program, [distances], 'int32', [], true) 
    if (verbose) logMean3d('isotropicDistances', isotropicDistances)

    return isotropicDistances as tf.Tensor3D
}

function anisotropicChebyshevDistancePass(
    distances: tf.Tensor3D,
    sweepAxis: Axis,
    sweepSign: Sign,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = distances.shape as [number, number, number]
    const program = new AnisotropicChebyshevDistancePass(shape, sweepAxis, sweepSign, maxDistance)
    const anisotropicDistances = runWebGLProgram(program, [distances], 'int32', [], true) 
    if (verbose) logMean3d('anisotropicDistances', anisotropicDistances)

    return anisotropicDistances as tf.Tensor3D
}

function extendedChebyshevDistancePass(
    distances: tf.Tensor3D,
    dominantAxis: Axis,
    dominantSign: Sign,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = distances.shape as [number, number, number]
    const program = new ExtendedChebyshevDistancePass(shape, dominantAxis, dominantSign, maxDistance)
    const extendedDistances = runWebGLProgram(program, [distances], 'int32', [], true) 
    if (verbose) logMean3d('extendedDistances', extendedDistances)

    return extendedDistances as tf.Tensor3D
}

/**
 * Builds an ordinary 3D Chebyshev distance map by applying symmetric 1D
 * transforms along x, then y, then z.
 */
export function computeIsotropicDistanceMap(
    shadowMap: tf.Tensor3D,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    if (verbose) logMean3d('shadowMap', shadowMap)
    const distances0d = initialChebyshevDistancePass(shadowMap, maxDistance)

    const distances1d = isotropicChebyshevDistancePass(distances0d, 'x', maxDistance)
    tf.dispose(distances0d)

    const distances2d = isotropicChebyshevDistancePass(distances1d, 'y', maxDistance)
    tf.dispose(distances1d)

    const distances3d = isotropicChebyshevDistancePass(distances2d, 'z', maxDistance)
    tf.dispose(distances2d)
    if (verbose) logMean3d('distanceMap', distances3d)

    return distances3d
}

/**
 * Builds one octant-oriented extended anisotropic distance map. The two
 * non-dominant axes are transformed first, then the dominant-axis pass extends
 * only along the ray direction.
 */
export function computeUnidirectionalDistanceMap(
    shadowMap: tf.Tensor3D,
    dominantAxis: Axis,
    directionOctant: Octant,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    if (verbose) logMean3d('shadowMap', shadowMap)
    const distances0d = initialChebyshevDistancePass(shadowMap, maxDistance)

    const sweepAxes =  ['x','y','z'].filter((axis) => axis !== dominantAxis) as [Axis, Axis]
    const sweepSigns = sweepAxes.map((sweepAxis) => su.getOctantSign(directionOctant, sweepAxis)) as [Sign, Sign]
    const dominantSign = su.getOctantSign(directionOctant, dominantAxis)

    const distances1d = anisotropicChebyshevDistancePass(distances0d, sweepAxes[0], sweepSigns[0], maxDistance)
    tf.dispose(distances0d)

    const distances2d = anisotropicChebyshevDistancePass(distances1d, sweepAxes[1], sweepSigns[1], maxDistance)
    tf.dispose(distances1d)

    const distances3d = extendedChebyshevDistancePass(distances2d, dominantAxis, dominantSign, maxDistance)
    tf.dispose(distances2d)
    if (verbose) logMean3d('distanceMap', distances3d)

    return distances3d
}

/**
 * Computes both directions of the same ray family from the same initial shadowMap
 * and keeps the shorter distance at each cell.
 */
export function computeBidirectionalDistanceMap(
    shadowMap: tf.Tensor3D,
    dominantAxis: Axis,
    directionOctant: Octant,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    if (verbose) logMean3d('shadowMap', shadowMap)
    const distances0d = initialChebyshevDistancePass(shadowMap, maxDistance)

    // Forward
    const sweepAxes =  ['x','y','z'].filter((axis) => axis !== dominantAxis) as [Axis, Axis]

    const forwardDominantSign = su.getOctantSign(directionOctant, dominantAxis)
    const forwardSweepSigns = sweepAxes.map((sweepAxis) => su.getOctantSign(directionOctant, sweepAxis)) as [Sign, Sign]

    const forwardDistances1d = anisotropicChebyshevDistancePass(distances0d, sweepAxes[0], forwardSweepSigns[0], maxDistance)

    const forwardDistances2d = anisotropicChebyshevDistancePass(forwardDistances1d, sweepAxes[1], forwardSweepSigns[1], maxDistance)
    tf.dispose(forwardDistances1d)

    const forwardDistances3d = extendedChebyshevDistancePass(forwardDistances2d, dominantAxis, forwardDominantSign, maxDistance)
    tf.dispose(forwardDistances2d)

    // Backward
    const backwardDominantSign = su.reverseSign(forwardDominantSign)
    const backwardSweepSigns = forwardSweepSigns.map((sweepSign) => su.reverseSign(sweepSign))

    const backwardDistances1d = anisotropicChebyshevDistancePass(distances0d, sweepAxes[0], backwardSweepSigns[0], maxDistance)
    tf.dispose(distances0d)

    const backwardDistances2d = anisotropicChebyshevDistancePass(backwardDistances1d, sweepAxes[1], backwardSweepSigns[1], maxDistance)
    tf.dispose(backwardDistances1d)

    const backwardDistances3d = extendedChebyshevDistancePass(backwardDistances2d, dominantAxis, backwardDominantSign, maxDistance)
    tf.dispose(backwardDistances2d)

    // Bidirectional
    const bidirectionalDistanceMap = tf.minimum(forwardDistances3d, backwardDistances3d)
    tf.dispose([forwardDistances3d, backwardDistances3d])
    if (verbose) logMean3d('distanceMap', bidirectionalDistanceMap)

    return bidirectionalDistanceMap as tf.Tensor3D
}


