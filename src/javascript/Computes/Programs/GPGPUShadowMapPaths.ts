
import * as tf from '@tensorflow/tfjs'
import * as su from '../../Utils/ShadowMapUtils'
import * as DistanceUtils from './GPGPUDistanceMap'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { stack3d } from './stack3d'
import { unstack3d } from './unstack3d'
import { minPool3d, maxPool3d, avgPool3d } from './pool3d'
import { type Sign, type Octant, type Axis } from '../../Utils/ShadowMapUtils'

/**
 * Directional shadow-map preprocessing for MIP distance maps.
 *
 * The shader stencils are written once in a canonical local cell space where
 * the ray advances through local +z. The TypeScript offset helpers below map
 * that canonical stencil to the requested dominant axis and octant.
 *
 * TensorFlow tensors are shaped/indexed as [z, y, x], while the GLSL snippets
 * use ivec3(x, y, z) for geometric offsets. All offset helpers therefore work
 * in z/y/x internally, then emit x/y/z GLSL arguments.
 *
 * The returned masks use 1 for rejected cells and 0 for cells that may still
 * contribute to the maximum intensity projection. Volumes are expected to be
 * normalized and non-negative; out-of-volume samples are treated as 0.0.
 */

/**
 * Emits GLSL ivec3 constructor arguments from an internal [z, y, x] tuple.
 */
function xyz(zyx: [number, number, number]): string
{
    return [zyx[2], zyx[1], zyx[0]].join(', ')
}

/**
 * Converts a canonical vertex-neighbor offset into physical tensor space.
 * Reversed axes flip offset signs because voxel addresses move across the
 * volume grid.
 */
function vertexOffset(
    x: number,
    y: number,
    z: number,
    axis: su.Axis,
    octant: su.Octant,
): string
{
    const { permute, reverse } = su.dominantAxisOctantToPermuteReverse(axis, octant)

    const zyx = su.applyPermutation([z, y, x], permute)

    for (const dimension of reverse) zyx[dimension] = -zyx[dimension]

    return xyz(zyx)
}

/**
 * Converts a canonical propagation offset for a pair of already-unstacked
 * slices. Any movement along the sweep axis is removed because the previous
 * sweep slice is supplied as a separate input tensor.
 */
function sliceOffset(
    x: number,
    y: number,
    z: number,
    axis: su.Axis,
    octant: su.Octant,
): string
{
    const { permute, reverse } = su.dominantAxisOctantToPermuteReverse(axis, octant)

    const zyx = su.applyPermutation([z, y, x], permute)

    for (const dimension of reverse) zyx[dimension] = -zyx[dimension]

    zyx[permute[0]] = 0

    return xyz(zyx)
}

/**
 * Converts a canonical cell-corner offset. Reversal mirrors a unit-cell corner,
 * so 0 becomes 1 and 1 becomes 0 instead of changing the sign.
 */
function cellOffset(
    x: number,
    y: number,
    z: number,
    axis: su.Axis,
    octant: su.Octant,
): string
{
    const { permute, reverse } = su.dominantAxisOctantToPermuteReverse(axis, octant)

    const zyx = su.applyPermutation([z, y, x], permute)

    for (const dimension of reverse) zyx[dimension] = 1 - zyx[dimension]

    return xyz(zyx)
}

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
 * Expands voxel intensities to a padded vertex lattice. Padding lets later
 * stencils address cell corners uniformly while clamped voxel sampling keeps
 * boundary values well-defined.
 */
class ComputeVertexValuesProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        shape: [number, number, number],
        axis: su.Axis = 'z',
        octant: su.Octant = '+++',
    ) {
        const [depth, height, width] = shape

        this.outputShape = [depth + 2, height + 2, width + 2]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool validVoxelCoords(ivec3 voxelCoords)
        {
            return 
                voxelCoords.x >= minCoords.x && voxelCoords.x <= maxCoords.x &&
                voxelCoords.y >= minCoords.y && voxelCoords.y <= maxCoords.y &&
                voxelCoords.z >= minCoords.z && voxelCoords.z <= maxCoords.z;
        }

        ivec3 outputCoords()
        {
            ivec3 vertexCoords = getOutputCoords();
            return ivec3(vertexCoords.z, vertexCoords.y, vertexCoords.x);
        }

        float voxelValueAt(ivec3 voxelCoords)
        {
            voxelCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(voxelCoords.z, voxelCoords.y, voxelCoords.x);
        }

        float vertexValueAt(ivec3 vertexCoords)
        {
            ivec3 voxelCoords = vertexCoords - 1;
            return voxelValueAt(voxelCoords);
        }

        void main()
        {
            setOutput(vertexValueAt(outputCoords()));
        }
        `
    }
}

/**
 * Reference fixed-point relaxation pass. It recomputes the whole volume until
 * information has propagated across the selected sweep length.
 *
 * This is slower than PropagateVertexMinmaxValuesProgram, but useful as a
 * conceptual reference because every iteration applies the same local stencil
 * everywhere and reads only the previous tensor.
 */
class IterateVertexMinmaxValuesProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        sliceShape: [number, number, number],
        axis: su.Axis = 'z',
        octant: su.Octant = '+++',
    ) {
        const [depth, height, width] = sliceShape

        this.outputShape = sliceShape
        this.userCode = `
        const float NEG_INF = -3.402823466e+38;

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool validVertexCoords(ivec3 vertexCoords)
        {
            return 
                vertexCoords.x >= minCoords.x && vertexCoords.x <= maxCoords.x &&
                vertexCoords.y >= minCoords.y && vertexCoords.y <= maxCoords.y &&
                vertexCoords.z >= minCoords.z && vertexCoords.z <= maxCoords.z;
        }

        ivec3 outputCoords()
        {
            ivec3 vertexCoords = getOutputCoords();
            return ivec3(vertexCoords.z, vertexCoords.y, vertexCoords.x);
        }

        float vertexMinmaxValueAt(ivec3 vertexCoords)
        {            
            return validVertexCoords(vertexCoords) ? 
                getA(vertexCoords.z, vertexCoords.y, vertexCoords.x) : NEG_INF;
        }

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        float computeVertexMinmaxValues(ivec3 vertexCoords)
        {
            float v111 = vertexMinmaxValueAt(vertexCoords + ivec3(${vertexOffset( 0,  0,  0, axis, octant)}));
            float v110 = vertexMinmaxValueAt(vertexCoords + ivec3(${vertexOffset( 0,  0, -1, axis, octant)}));
            float v100 = vertexMinmaxValueAt(vertexCoords + ivec3(${vertexOffset( 0, -1, -1, axis, octant)}));
            float v010 = vertexMinmaxValueAt(vertexCoords + ivec3(${vertexOffset(-1,  0, -1, axis, octant)}));
            float v000 = vertexMinmaxValueAt(vertexCoords + ivec3(${vertexOffset(-1, -1, -1, axis, octant)}));

            float minValue = min4(v000, v010, v100, v110);

            return max(v111, minValue);
        }

        void main()
        {
            setOutput(computeVertexMinmaxValues(outputCoords()));
        }
        `
    }
}

/**
 * One dynamic-programming propagation step for a single sweep slice.
 *
 * A is the current raw slice. B is the already-propagated previous slice in
 * the requested direction. The shader keeps the current vertex value unless
 * every incoming route has already seen a larger guaranteed minimum.
 */
class PropagateVertexMinmaxValuesProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: [number, number, number]
    userCode: string
    packedInputs = false
    packedOutput = false
    customUniforms = [{ name: 'slice', type: 'int' as const }]

    constructor(
        sliceShape: [number, number, number],
        axis: su.Axis = 'z',
        octant: su.Octant = '+++',
    ) {
        const [depth, height, width] = sliceShape

        this.outputShape = sliceShape
        this.userCode = `
        const float NEG_INF = -3.402823466e+38;

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool validSliceCoords(ivec3 sliceCoords)
        {
            return 
                ${(width  > 1) ? 'sliceCoords.x >= minCoords.x && sliceCoords.x <= maxCoords.x' : 'true'} &&
                ${(height > 1) ? 'sliceCoords.y >= minCoords.y && sliceCoords.y <= maxCoords.y' : 'true'} &&
                ${(depth  > 1) ? 'sliceCoords.z >= minCoords.z && sliceCoords.z <= maxCoords.z' : 'true'};
        }

        ivec3 outputCoords()
        {
            ivec3 sliceCoords = getOutputCoords();
            return ivec3(sliceCoords.z, sliceCoords.y, sliceCoords.x);
        }

        float currentSliceAt(ivec3 sliceCoords)
        {
            return getA(sliceCoords.z, sliceCoords.y, sliceCoords.x);
        }

        float previousSliceAt(ivec3 sliceCoords)
        {            
            return validSliceCoords(sliceCoords) ? 
                getB(sliceCoords.z, sliceCoords.y, sliceCoords.x) : NEG_INF;
        }

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        float computeVertexMinmaxValues(ivec3 sliceCoords)
        {
            float v111 =  currentSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  0,  0, axis, octant)}));
            float v110 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  0, -1, axis, octant)}));
            float v100 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0, -1, -1, axis, octant)}));
            float v010 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1,  0, -1, axis, octant)}));
            float v000 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1, -1, -1, axis, octant)}));

            float minValue = min4(v000, v010, v100, v110);

            return max(v111, minValue);
        }

        void main()
        {
            setOutput(computeVertexMinmaxValues(outputCoords()));
        }
        `
    }
}

/**
 * Classifies vertices for one oriented ray class.
 *
 * A contains the original vertex values. B contains propagated minmax values.
 * A vertex is shadowed when its value is within tolerance of the minValues
 * already available from the previous face.
 */
class ComputeVertexShadowsProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: [number, number, number]
    userCode: string
    packedInputs = false
    packedOutput = false
    customUniforms = [{ name: 'tolerance', type: 'float' as const }]

    constructor(
        shape: [number, number, number],
        axis: su.Axis = 'z',
        octant: su.Octant = '+++'
    ) {
        const [depth, height, width] = shape
    
        this.outputShape = shape
        this.userCode = `
        const float NEG_INF = -3.402823466e+38;

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool validVertexCoords(ivec3 vertexCoords)
        {
            return 
                vertexCoords.x >= minCoords.x && vertexCoords.x <= maxCoords.x &&
                vertexCoords.y >= minCoords.y && vertexCoords.y <= maxCoords.y &&
                vertexCoords.z >= minCoords.z && vertexCoords.z <= maxCoords.z;
        }

        ivec3 outputCoords()
        {
            ivec3 vertexCoords = getOutputCoords();
            return ivec3(vertexCoords.z, vertexCoords.y, vertexCoords.x);
        }

        float vertexValueAt(ivec3 vertexCoords)
        {
            return getA(vertexCoords.z, vertexCoords.y, vertexCoords.x);
        }

        float vertexMinmaxAt(ivec3 vertexCoords)
        {
            return validVertexCoords(vertexCoords) ? 
                getB(vertexCoords.z, vertexCoords.y, vertexCoords.x) : NEG_INF;
        }

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        bool computeVertexShadow(ivec3 vertexCoords)
        {
            float v111 =  vertexValueAt(vertexCoords + ivec3(${vertexOffset( 0,  0,  0, axis, octant)}));
            float v110 = vertexMinmaxAt(vertexCoords + ivec3(${vertexOffset( 0,  0, -1, axis, octant)}));
            float v100 = vertexMinmaxAt(vertexCoords + ivec3(${vertexOffset( 0, -1, -1, axis, octant)}));
            float v010 = vertexMinmaxAt(vertexCoords + ivec3(${vertexOffset(-1,  0, -1, axis, octant)}));
            float v000 = vertexMinmaxAt(vertexCoords + ivec3(${vertexOffset(-1, -1, -1, axis, octant)}));

            float minValue = min4(v000, v010, v100, v110);
            float margin = v111 - minValue;

            return (margin <= tolerance);
        }

        void main()
        {
            setOutput(float(computeVertexShadow(outputCoords())));
        }
        `
    }
}

/**
 * Builds the reverse-pass hole mask from a forward cell mask.
 *
 * During the backward pass, cells already rejected by the forward pass should
 * not act as solid occluders. The hole tensor zeros those vertex values before
 * the reverse propagation.
 */
class ComputeVertexHolesProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        shape: [number, number, number],
        axis: su.Axis = 'z',
        octant: su.Octant = '+++',
    ) {
        const [depth, height, width] = shape

        this.outputShape = [depth + 1, height + 1, width + 1]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        ivec3 outputCoords()
        {
            ivec3 vertexCoords = getOutputCoords();
            return ivec3(vertexCoords.z, vertexCoords.y, vertexCoords.x);
        }

        bool cellShadowAt(ivec3 cellCoords)
        {
            cellCoords = clamp(cellCoords, minCoords, maxCoords);
            return getA(cellCoords.z, cellCoords.y, cellCoords.x) > 0.5;
        }

        bool computeVertexHole(ivec3 vertexCoords)
        {
            return cellShadowAt(vertexCoords - ivec3(${cellOffset(0, 0, 0, axis, octant)}));
        }

        void main()
        {            
            setOutput(float(computeVertexHole(outputCoords())));
        }
        `
    }
}

/**
 * Promotes vertex shadows to cell shadows.
 *
 * A cell is rejected only when all relevant corners for this oriented ray class
 * are shadowed. This keeps the final culling mask conservative.
 */
class ComputeCellShadowsProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        shape: [number, number, number],
        axis: su.Axis = 'z',
        octant: su.Octant = '+++',
    ) {
        const [depth, height, width] = shape

        this.outputShape = [depth - 1, height - 1, width - 1]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});
            
        ivec3 outputCoords()
        {
            ivec3 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        bool vertexShadowAt(ivec3 vertexCoords)
        {
            vertexCoords = clamp(vertexCoords, minCoords, maxCoords);
            return getA(vertexCoords.z, vertexCoords.y, vertexCoords.x) > 0.5;
        }

        bool computeCellShadow(ivec3 cellCoords)
        {
            return (
                vertexShadowAt(cellCoords + ivec3(${cellOffset(1, 1, 1, axis, octant)})) &&
                vertexShadowAt(cellCoords + ivec3(${cellOffset(1, 0, 1, axis, octant)})) &&
                vertexShadowAt(cellCoords + ivec3(${cellOffset(0, 1, 1, axis, octant)})) &&
                vertexShadowAt(cellCoords + ivec3(${cellOffset(0, 0, 1, axis, octant)})) &&
                vertexShadowAt(cellCoords + ivec3(${cellOffset(1, 1, 0, axis, octant)})) &&
                vertexShadowAt(cellCoords + ivec3(${cellOffset(1, 0, 0, axis, octant)})) &&
                vertexShadowAt(cellCoords + ivec3(${cellOffset(0, 1, 0, axis, octant)})) 
            );
        }

        void main()
        {
            setOutput(float(computeCellShadow(outputCoords())));
        }
        `
    }
}


function computeVertexValues(
    volume: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = volume.shape as [number, number, number]
    const program = new ComputeVertexValuesProgram(shape, axis, octant)
    const vertexValues = runWebGLProgram(program, [volume], 'float32', [], true) 
    if (verbose) logMean3d('vertexValues', vertexValues)

    return vertexValues as tf.Tensor3D
}

/**
 * Slow reference implementation of directional minmax propagation.
 *
 * This is the Jacobi-style variant: each pass writes a new tensor from the old
 * tensor, so information advances by one stencil edge per pass.
 */
function iterateVertexMinmaxValues(
    volume: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    iterations: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = volume.shape as [number, number, number]
    const propagate = new IterateVertexMinmaxValuesProgram(shape, axis, octant)

    let prev = volume.clone() 

    for (let i = 0; i < iterations; i += 1)
    {
        const next = runWebGLProgram(propagate, [prev], 'float32', [], true)
        tf.dispose(prev)
        prev = next  as tf.Tensor3D
    }

    const vertexMinmaxValues = prev
    if (verbose) logMean3d('vertexMinmaxValues', vertexMinmaxValues)

    return vertexMinmaxValues as tf.Tensor3D
}

/**
 * Propagates directional minmax values one slice at a time along the selected
 * axis and octant. This is the fast path used by the public shadow-map helpers:
 * each slice sees the already-propagated slice behind it in the sweep order.
 */
function propagateVertexMinmaxValues(
    vertexValues: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    verbose: boolean = false
): tf.Tensor3D
{
    const dimension = su.axisToDimension(axis)
    const sign = su.getOctantSign(octant, axis)
    const backwards = sign === '-'

    const slices = unstack3d(vertexValues, dimension)
    const shape = slices[0].shape as [number, number, number]
    const propagate = new PropagateVertexMinmaxValuesProgram(shape, axis, octant)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length // 0 : slices.length - 1
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step)
    {
        const next = runWebGLProgram(propagate, [slices[i], slices[i-step]], 'float32', [[i]], true)
        tf.dispose(slices[i])
        slices[i] = next
    }

    const vertexMinmaxValues = stack3d(slices, dimension) 
    tf.dispose(slices)
    if (verbose) logMean3d('vertexMinmaxValues', vertexMinmaxValues)

    return vertexMinmaxValues as tf.Tensor3D
}

/**
 * Converts propagated minmax values into a binary vertex shadow mask.
 */
function computeVertexShadows(
    vertexValues: tf.Tensor3D,
    vertexMinmaxValues: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const program = new ComputeVertexShadowsProgram(vertexValues.shape, axis, octant)
    const vertexShadows = runWebGLProgram(program, [vertexValues, vertexMinmaxValues], 'bool', [[tolerance]], true) 
    if (verbose) logMean3d('vertexShadows', vertexShadows)

    return vertexShadows as tf.Tensor3D
}

/**
 * Converts a binary vertex shadow mask into a binary cell shadow mask.
 */
function computeCellShadows(
    vertexShadows: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    verbose: boolean = false
): tf.Tensor3D
{
    const program = new ComputeCellShadowsProgram(vertexShadows.shape, axis, octant)
    const cellShadows = runWebGLProgram(program, [vertexShadows], 'bool', [], true) 
    if (verbose) logMean3d('cellShadows', cellShadows)

    return cellShadows as tf.Tensor3D
}

/**
 * Converts forward cell shadows into vertex holes for the reverse sweep.
 */
function computeVertexHoles(
    cellShadows: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    verbose: boolean = false
): tf.Tensor3D
{
    const program = new ComputeVertexHolesProgram(cellShadows.shape, axis, octant)
    const vertexHoles = runWebGLProgram(program, [cellShadows], 'bool', [], true) 
    if (verbose) logMean3d('vertexHoles', vertexHoles)

    return vertexHoles as tf.Tensor3D
}

/**
 * Computes one directional conservative cell-rejection mask.
 */
export function computeUnidirectionalShadowMap(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    directionOctant: Octant,
    errorTolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const vertexValues = computeVertexValues(volume, dominantAxis, directionOctant, verbose)
    const vertexMinmaxValues = propagateVertexMinmaxValues(vertexValues, dominantAxis, directionOctant, verbose)
    const vertexShadows = computeVertexShadows(vertexValues, vertexMinmaxValues, dominantAxis, directionOctant, errorTolerance, verbose)
    tf.dispose([vertexValues, vertexMinmaxValues])

    const cellShadows = computeCellShadows(vertexShadows, dominantAxis, directionOctant, verbose)
    tf.dispose(vertexShadows)
    if (verbose) logMean3d('cellShadows', cellShadows)
        
    if (blockSize === 1) 
        return cellShadows as tf.Tensor3D

    const blockShadows = minPool3d(cellShadows, blockSize, blockSize, 0, 'ceil') 
    tf.dispose(cellShadows)
    if (verbose) logMean3d('blockShadows', blockShadows)
        
    return blockShadows as tf.Tensor3D
}

/**
 * Computes the conservative mask for both directions of the same ray family.
 */
export function computeBidirectionalShadowMap(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    directionOctant: Octant,
    errorTolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    // Forward 
    const forwardOctant = directionOctant
    const forwardVertexValues = computeVertexValues(volume, dominantAxis, directionOctant, verbose)
    const forwardVertexMinmaxValues = propagateVertexMinmaxValues(forwardVertexValues, dominantAxis, forwardOctant, verbose)
    const forwardVertexShadows = computeVertexShadows(forwardVertexValues, forwardVertexMinmaxValues, dominantAxis, forwardOctant, errorTolerance, verbose)
    tf.dispose(forwardVertexMinmaxValues)

    const forwardCellShadows = computeCellShadows(forwardVertexShadows, dominantAxis, forwardOctant, false)
    tf.dispose(forwardVertexShadows)
    if (verbose) logMean3d('forwardCellShadows', forwardCellShadows)

    // Backward 
    const backwardOctant = su.reverseOctant(forwardOctant)
    const backwardVertexHoles = computeVertexHoles(forwardCellShadows, dominantAxis, backwardOctant, verbose)
    const backwardVertexValues = tf.where(backwardVertexHoles, 0, forwardVertexValues) 
    tf.dispose([backwardVertexHoles, forwardVertexValues])

    const backwardVertexMinmaxValues = propagateVertexMinmaxValues(backwardVertexValues, dominantAxis, backwardOctant, verbose)
    const backwardVertexShadows = computeVertexShadows(backwardVertexValues, backwardVertexMinmaxValues, dominantAxis, backwardOctant, errorTolerance, verbose)
    tf.dispose([backwardVertexValues, backwardVertexMinmaxValues])

    const backwardCellShadows = computeCellShadows(backwardVertexShadows, dominantAxis, backwardOctant, false)
    tf.dispose(backwardVertexShadows)
    if (verbose) logMean3d('backwardCellShadows', backwardCellShadows)

    // Bidirectional 
    const bidirectionalCellShadows = tf.logicalOr(forwardCellShadows, backwardCellShadows) as tf.Tensor3D
    tf.dispose([forwardCellShadows, backwardCellShadows])
    if (verbose) logMean3d('bidirectionalCellShadows', bidirectionalCellShadows)

    if (blockSize === 1) 
        return bidirectionalCellShadows as tf.Tensor3D

    const bidirectionalBlockShadows = minPool3d(bidirectionalCellShadows, blockSize, blockSize, 0, 'ceil') 
    tf.dispose(bidirectionalCellShadows)
    if (verbose) logMean3d('bidirectionalBlockShadows', bidirectionalBlockShadows)

    return bidirectionalBlockShadows as tf.Tensor3D
}

/**
 * Computes the conservative mask first, then measures a unidirectional distance
 * map for the requested ray family.
 */
export function computeUnidirectionalDistanceMap(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    directionOctant: Octant,
    errorTolerance: number,
    blockSize: number,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shadowMask = computeBidirectionalShadowMap(volume, dominantAxis, directionOctant, errorTolerance, blockSize, verbose)
    const distanceMap = DistanceUtils.computeUnidirectionalDistanceMap(shadowMask, dominantAxis, directionOctant, maxDistance, verbose)
    tf.dispose(shadowMask)

    return distanceMap
}

/**
 * Computes the bidirectional conservative mask before measuring distance to the
 * nearest non-rejected seed in either direction of the ray family.
 */
export function computeBidirectionalDistanceMap(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    directionOctant: Octant,
    errorTolerance: number,
    blockSize: number,
    maxDistance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shadowMask = computeBidirectionalShadowMap(volume, dominantAxis, directionOctant, errorTolerance, blockSize, verbose)
    const distanceMap = DistanceUtils.computeBidirectionalDistanceMap(shadowMask, dominantAxis, directionOctant, maxDistance, verbose)
    tf.dispose(shadowMask)

    return distanceMap
}


/**
 * Alternative block-first variant. It pools the input value bounds first, then
 * computes the bidirectional mask at block resolution.
 */
export function computeBidirectionalShadowMap2(
    volume: tf.Tensor3D,
    axis: Axis,
    octant: Octant,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    // Forward
    const forwardOctant = octant
    const forwardMinVertexValues = minPool3d(volume, blockSize+1, blockSize, 0, 'floor')
    const forwardMaxVertexValues = maxPool3d(volume, blockSize+1, blockSize, 0, 'floor')

    const forwardVertexMinmaxValues = propagateVertexMinmaxValues(forwardMinVertexValues, axis, forwardOctant, verbose)
    const forwardVertexShadows = computeVertexShadows(forwardMaxVertexValues, forwardVertexMinmaxValues, axis, forwardOctant, tolerance, verbose)
    tf.dispose(forwardVertexMinmaxValues)

    const forwardCellShadows = computeCellShadows(forwardVertexShadows, axis, forwardOctant, false)
    tf.dispose(forwardVertexShadows)
    if (verbose) logMean3d('forwardCellShadows', forwardCellShadows)

    // Backward 
    const backwardOctant = su.reverseOctant(forwardOctant)
    const backwardVertexHoles = computeVertexHoles(forwardCellShadows, axis, backwardOctant, verbose)

    const backwardMinVertexValues = tf.where(backwardVertexHoles, 0, forwardMinVertexValues) 
    const backwardMaxVertexValues = tf.where(backwardVertexHoles, 0, forwardMaxVertexValues) 
    tf.dispose([backwardVertexHoles, forwardMinVertexValues, forwardMaxVertexValues])

    const backwardVertexMinmaxValues = propagateVertexMinmaxValues(backwardMinVertexValues, axis, backwardOctant, verbose)
    tf.dispose(backwardMinVertexValues)

    const backwardVertexShadows = computeVertexShadows(backwardMaxVertexValues, backwardVertexMinmaxValues, axis, backwardOctant, tolerance, verbose)
    tf.dispose([backwardMaxVertexValues, backwardVertexMinmaxValues])

    const backwardCellShadows = computeCellShadows(backwardVertexShadows, axis, backwardOctant, false)
    tf.dispose(backwardVertexShadows)
    if (verbose) logMean3d('backwardCellShadows', backwardCellShadows)

    // Bidirectional 
    const bidirectionalCellShadows = tf.logicalOr(forwardCellShadows, backwardCellShadows)
    tf.dispose([forwardCellShadows, backwardCellShadows])
    if (verbose) logMean3d('bidirectionalCellShadows', bidirectionalCellShadows)

    return bidirectionalCellShadows as tf.Tensor3D
}
