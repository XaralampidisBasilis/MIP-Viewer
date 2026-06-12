

import * as tf from '@tensorflow/tfjs'
import * as su from '../../Utils/ShadowMapUtils'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { minimum3dPacked } from './minimum3dPacked'
import { type Sign, type Octant, type Axis } from '../../Utils/ShadowMapUtils'

/**
 * This function returns the order of the packed octants that the packed
 *  gpgpu programs bellow generate given as input the dominant axis and sign
 */
function packedOctantsFromSignAxis(sign: Sign, axis: Axis): [Octant, Octant, Octant, Octant]
{
    const PACKED_OCTANTS_FROM_SIGN_AXIS: Record<string, [Octant, Octant, Octant, Octant]> = 
    {
        "+x" : ['+++', '+-+', '++-', '+--'] , "-x" : ['---', '-+-', '--+', '-++'] ,
        "+y" : ['+++', '-++', '++-', '-+-'] , "-y" : ['---', '+--', '--+', '+-+'] ,
        "+z" : ['+++', '+-+', '-++', '--+'] , "-z" : ['---', '-+-', '+--', '++-'] ,
    }

    const key = `${sign}${axis}`
    const octants = PACKED_OCTANTS_FROM_SIGN_AXIS[key]

    return [...octants] as [Octant, Octant, Octant, Octant]
}

function unpackTensorFromAxisOctant(
    tensor: tf.Tensor5D,
    dominantAxis: Axis,
    directionOctant: Octant,
): tf.Tensor3D
{
    const dominantSign = su.getOctantSign(directionOctant, dominantAxis)
    const octants = packedOctantsFromSignAxis(dominantSign, dominantAxis)
    const index = octants.findIndex((octant) => octant === directionOctant)
    const row = Math.floor(index / 2);
    const col = index % 2;

    return tf.tidy(() => tensor.slice([0, 0, 0, row, col], [-1, -1, -1, 1, 1]).squeeze([3, 4]))
}

/**
 * Logs the mean over spatial axes without downloading the full tensor.
 */
function logMean3d(name: string, tensor: tf.Tensor): void
{
    console.log(name, tf.tidy(() => tensor.mean([0,1,2]).dataSync()))
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

class SetupChebyshevDistancePass implements GPGPUProgram 
{
    variableNames = ['A'] 
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        shape: [number, number, number, 2, 2],
        maxDistance: number,
    ) {
        this.outputShape = shape
        this.userCode = `
        ivec3 outputCoords() 
        {
            ivec5 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        vec4 cellShadowsAt(ivec3 coords) 
        {
            return step(0.5, getA(coords.z, coords.y, coords.x, 0, 0));
        }

        void main() 
        {
            vec4 cellShadows = cellShadowsAt(outputCoords());
            vec4 cellDistances = mix(vec4(0), vec4(${maxDistance}), cellShadows);
            
            setOutput(cellDistances);
        }
        `
    }
}

class IsotropicChebyshevDistancePass implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        shape: [number, number, number, 2, 2],
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
            ivec5 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        ivec4 sampleChebyshevDistancesAt(ivec3 cellCoords)
        {
            return ivec4(getA(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0));
        }
    
        void main()
        {
            ivec3 outCoords = outputCoords();
            ivec3 sampleCoords = outCoords;

            ivec4 minDistances = sampleChebyshevDistancesAt(sampleCoords);

            if (all(equal(minDistances, ivec4(0))))
            {
                setOutput(vec4(minDistances));
                return;
            }

            for (int radius = 1; radius <= maxRadius; ++radius)
            {
                ivec4 radius4 = ivec4(radius);

                sampleCoords.${sweepAxis} = outCoords.${sweepAxis} - radius;
                
                if (insideAxis(sampleCoords))
                {
                    ivec4 sampleDistances = sampleChebyshevDistancesAt(sampleCoords);
                    ivec4 candidateDistances = max(sampleDistances, radius4);

                    minDistances = min(minDistances, candidateDistances);

                    if (all(lessThanEqual(minDistances, radius4))) 
                    {
                        break;
                    }
                }

                sampleCoords.${sweepAxis} = outCoords.${sweepAxis} + radius;

                if (insideAxis(sampleCoords))
                {
                    ivec4 sampleDistances = sampleChebyshevDistancesAt(sampleCoords);
                    ivec4 candidateDistances = max(sampleDistances, radius4);

                    minDistances = min(minDistances, candidateDistances);

                    if (all(lessThanEqual(minDistances, radius4)))
                    {
                        break;
                    }
                }
            }

            setOutput(vec4(minDistances));
        }
        `
    }
}

class AnisotropicChebyshevDistancePass implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        shape: [number, number, number, 2, 2],
        sweepAxis: Axis,
        sweepSigns: [Sign, Sign, Sign, Sign],
        maxDistance: number,
    ) {
        const channels = (sign: Sign) => 
        'rgba'.split('').filter((_, i) => sweepSigns[i] === sign).join('')
        
        const negChannels = channels('-')
        const posChannels = channels('+')

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
            ivec5 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        ivec4 sampleChebyshevDistancesAt(ivec3 cellCoords)
        {
            return ivec4(getA(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0));
        }

        void main()
        {
            ivec3 outCoords = outputCoords();
            ivec3 sampleCoords = outCoords;

            ivec4 minDistances = sampleChebyshevDistancesAt(sampleCoords);

            if (all(equal(minDistances, ivec4(0))))
            {
                setOutput(vec4(minDistances));
                return;
            }

            for (int radius = 1; radius <= maxRadius; ++radius)
            {
                sampleCoords.${sweepAxis} = outCoords.${sweepAxis} - radius;
                
                if (outsideAxis(sampleCoords))
                {
                    break;
                }

                ivec4 radius4 = ivec4(radius);
                ivec4 sampleDistances = sampleChebyshevDistancesAt(sampleCoords);
                ivec4 candidateDistances = max(sampleDistances, radius4);

                minDistances.${negChannels} = min(minDistances.${negChannels}, candidateDistances.${negChannels});

                if (all(lessThanEqual(minDistances.${negChannels}, radius4.${negChannels}))) 
                {
                    break;
                }
            }

            for (int radius = 1; radius <= maxRadius; ++radius)
            {
                sampleCoords.${sweepAxis} = outCoords.${sweepAxis} + radius;

                if (outsideAxis(sampleCoords))
                {
                    break;
                }

                ivec4 radius4 = ivec4(radius);
                ivec4 sampleDistances = sampleChebyshevDistancesAt(sampleCoords);
                ivec4 candidateDistances = max(sampleDistances, radius4);

                minDistances.${posChannels} = min(minDistances.${posChannels}, candidateDistances.${posChannels});

                if (all(lessThanEqual(minDistances.${posChannels}, radius4.${posChannels}))) 
                {
                    break;
                }
            }

            setOutput(vec4(minDistances));
        }
        `
    }
}

class ExtendedChebyshevDistancePass implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        shape: [number, number, number, 2, 2],
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
            ivec5 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        ivec4 sampleChebyshevDistancesAt(ivec3 cellCoords)
        {
            return ivec4(getA(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0));
        }

        ivec4 getCandidateDistances(ivec4 sampleDistances, int radius)
        {
            return ivec4(
                sampleDistances.r <= radius ? radius : ${maxDistance},
                sampleDistances.g <= radius ? radius : ${maxDistance},
                sampleDistances.b <= radius ? radius : ${maxDistance},
                sampleDistances.a <= radius ? radius : ${maxDistance}
            );
        }

        void main() 
        {
            ivec3 outCoords = outputCoords();
            ivec3 sampleCoords = outCoords;

            ivec4 minDistances = ivec4(${maxDistance});

            for (int radius = 0; radius <= maxRadius; ++radius) 
            {
                sampleCoords.${dominantAxis} = outCoords.${dominantAxis} ${dominantSign} radius;

                if (outsideAxis(sampleCoords)) 
                {
                    break;
                }

                ivec4 radius4 = ivec4(radius);
                ivec4 sampleDistances = sampleChebyshevDistancesAt(sampleCoords);
                ivec4 candidateDistances = getCandidateDistances(sampleDistances, radius);
                
                minDistances = min(minDistances, candidateDistances); 
                
                if (all(lessThanEqual(minDistances, radius4))) 
                {
                    break;
                }
            }

            setOutput(vec4(minDistances));
        }
        `
    }
}

function setupChebyshevDistancePass(
    cellShadows: tf.Tensor5D,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = cellShadows.shape as [number, number, number, 2, 2]
    const program = new SetupChebyshevDistancePass(shape, maxDistance)
    const initialDistances = runWebGLProgram(program, [cellShadows], 'int32', [], true) 
    if (verbose) logMean3d('initialDistances', initialDistances)

    return initialDistances as tf.Tensor5D
}

function isotropicChebyshevDistancePass(
    distances: tf.Tensor5D,
    sweepAxis: Axis,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = distances.shape as [number, number, number, 2, 2]
    const program = new IsotropicChebyshevDistancePass(shape, sweepAxis, maxDistance)
    const isotropicDistances = runWebGLProgram(program, [distances], 'int32', [], true) 
    if (verbose) logMean3d('isotropicDistances', isotropicDistances)

    return isotropicDistances as tf.Tensor5D
}

function anisotropicChebyshevDistancePass(
    distances: tf.Tensor5D,
    sweepAxis: Axis,
    sweepSigns: [Sign, Sign, Sign, Sign],
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = distances.shape as [number, number, number, 2, 2]
    const program = new AnisotropicChebyshevDistancePass(shape, sweepAxis, sweepSigns, maxDistance)
    const anisotropicDistances = runWebGLProgram(program, [distances], 'int32', [], true) 
    if (verbose) logMean3d('anisotropicDistances', anisotropicDistances)

    return anisotropicDistances as tf.Tensor5D
}

function extendedChebyshevDistancePass(
    distances: tf.Tensor5D,
    dominantAxis: Axis,
    dominantSign: Sign,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = distances.shape as [number, number, number, 2, 2]
    const program = new ExtendedChebyshevDistancePass(shape, dominantAxis, dominantSign, maxDistance)
    const extendedDistances = runWebGLProgram(program, [distances], 'int32', [], true) 
    if (verbose) logMean3d('extendedDistances', extendedDistances)

    return extendedDistances as tf.Tensor5D
}

export function computeIsotropicDistanceMaps(
    mask: tf.Tensor5D,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor5D
{
    const distance0d = setupChebyshevDistancePass(mask, maxDistance, verbose)

    const distances1d = isotropicChebyshevDistancePass(distance0d, 'x', maxDistance, verbose)
    tf.dispose(distance0d)

    const distances2d = isotropicChebyshevDistancePass(distances1d, 'y', maxDistance, verbose)
    tf.dispose(distances1d)

    const distances3d = isotropicChebyshevDistancePass(distances2d, 'z', maxDistance, verbose)
    tf.dispose(distances2d)

    return distances3d
}

export function computeBidirectionalDistanceMaps(
    mask: tf.Tensor5D,
    dominantAxis: Axis,
    dominantSign: Sign,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor5D
{

    const sweepAxes =  ['x','y','z'].filter((axis) => axis !== dominantAxis) as [Axis, Axis]
    const sweepOctants = packedOctantsFromSignAxis(dominantSign, dominantAxis)

    const distances0d = setupChebyshevDistancePass(mask, maxDistance)

    // Forward
    const forwardDominantSign = dominantSign
    const forwardSweepSigns = sweepAxes.map((axis) => sweepOctants.map((octant) => su.getOctantSign(octant, axis)) as [Sign, Sign, Sign, Sign]) 

    const forwardDistances1d = anisotropicChebyshevDistancePass(distances0d, sweepAxes[0], forwardSweepSigns[0], maxDistance)

    const forwardDistances2d = anisotropicChebyshevDistancePass(forwardDistances1d, sweepAxes[1], forwardSweepSigns[1], maxDistance)
    tf.dispose(forwardDistances1d)

    const forwardDistances3d = extendedChebyshevDistancePass(forwardDistances2d, dominantAxis, forwardDominantSign, maxDistance)
    tf.dispose(forwardDistances2d)
    if (verbose) logMean3d('forwardDistanceMap', forwardDistances3d)

    // Backward
    const backwardDominantSign = su.reverseSign(forwardDominantSign)
    const backwardSweepSigns = forwardSweepSigns.map((signs) => signs.map((sign) => su.reverseSign(sign)) as [Sign, Sign, Sign, Sign])

    const backwardDistances1d = anisotropicChebyshevDistancePass(distances0d, sweepAxes[0], backwardSweepSigns[0], maxDistance)
    tf.dispose(distances0d)

    const backwardDistances2d = anisotropicChebyshevDistancePass(backwardDistances1d, sweepAxes[1], backwardSweepSigns[1], maxDistance)
    tf.dispose(backwardDistances1d)

    const backwardDistances3d = extendedChebyshevDistancePass(backwardDistances2d, dominantAxis, backwardDominantSign, maxDistance)
    tf.dispose(backwardDistances2d)
    if (verbose) logMean3d('backwardDistanceMap', backwardDistances3d)

    // Bidirectional
    const bidirectionalDistanceMap = minimum3dPacked(forwardDistances3d, backwardDistances3d)
    tf.dispose([forwardDistances3d, backwardDistances3d])
    if (verbose) logMean3d('bidirectionalDistanceMap', bidirectionalDistanceMap)

    return bidirectionalDistanceMap as tf.Tensor5D
}

export function computeUnidirectionalDistanceMaps(
    mask: tf.Tensor5D,
    dominantAxis: Axis,
    dominantSign: Sign,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor5D
{
    const sweepOctants = packedOctantsFromSignAxis(dominantSign, dominantAxis)
    const sweepAxes =  ['x','y','z'].filter((axis) => axis !== dominantAxis) as [Axis, Axis]
    const sweepSigns = sweepAxes.map((axis) => sweepOctants.map((octant) => su.getOctantSign(octant, axis)) as [Sign, Sign, Sign, Sign]) 

    const distance0d = setupChebyshevDistancePass(mask, maxDistance, verbose)

    const distances1d = anisotropicChebyshevDistancePass(distance0d, sweepAxes[0], sweepSigns[0], maxDistance, verbose)
    tf.dispose(distance0d)

    const distances2d = anisotropicChebyshevDistancePass(distances1d, sweepAxes[1], sweepSigns[1], maxDistance, verbose)
    tf.dispose(distances1d)

    const distances3d = extendedChebyshevDistancePass(distances2d, dominantAxis, dominantSign, maxDistance, verbose)
    tf.dispose(distances2d)

    return distances3d
}

export function computeIsotropicDistanceMap(
    mask: tf.Tensor5D,
    dominantAxis: Axis,
    directionOctant: Octant,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const distanceMaps = computeIsotropicDistanceMaps(mask, maxDistance, verbose)
    const distanceMap = unpackTensorFromAxisOctant(distanceMaps, dominantAxis, directionOctant)
    tf.dispose(distanceMaps)

    return distanceMap as tf.Tensor3D
}

export function computeBidirectionalDistanceMap(
    mask: tf.Tensor5D,
    dominantAxis: Axis,
    directionOctant: Octant,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const dominantSign = su.getOctantSign(directionOctant, dominantAxis)
    const distanceMaps = computeBidirectionalDistanceMaps(mask, dominantAxis, dominantSign, maxDistance, verbose)
    const distanceMap = unpackTensorFromAxisOctant(distanceMaps, dominantAxis, directionOctant)
    tf.dispose(distanceMaps)

    return distanceMap as tf.Tensor3D
}

export function computeUnidirectionalDistanceMap(
    mask: tf.Tensor5D,
    dominantAxis: Axis,
    directionOctant: Octant,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const dominantSign = su.getOctantSign(directionOctant, dominantAxis)
    const distanceMaps = computeUnidirectionalDistanceMaps(mask, dominantAxis, dominantSign, maxDistance, verbose)
    const distanceMap = unpackTensorFromAxisOctant(distanceMaps, dominantAxis, directionOctant)
    tf.dispose(distanceMaps)

    return distanceMap as tf.Tensor3D
}


