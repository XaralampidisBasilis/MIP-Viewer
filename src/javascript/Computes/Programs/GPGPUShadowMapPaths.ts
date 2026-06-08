
import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { stack3d } from './stack3d'
import { unstack3d } from './unstack3d'
import { minPool3d, maxPool3d, avgPool3d } from './pool3d'
import * as su from '../../Utils/ShadowMapUtils'

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
function voxelOffset(
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
 * Reference fixed-point relaxation pass. It recomputes the whole volume until
 * information has propagated across the selected sweep length.
 *
 * This is slower than PropagateVertexMinmaxProgram, but useful as a conceptual
 * reference because every iteration applies the same local stencil everywhere.
 */
class IterateVertexMinmaxProgram implements GPGPUProgram
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
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool insideVolume(ivec3 voxelCoords)
        {
            return 
                voxelCoords.x >= minCoords.x && voxelCoords.x <= maxCoords.x &&
                voxelCoords.y >= minCoords.y && voxelCoords.y <= maxCoords.y &&
                voxelCoords.z >= minCoords.z && voxelCoords.z <= maxCoords.z;
        }

        ivec3 outputCoords()
        {
            ivec3 voxelCoords = getOutputCoords();
            return ivec3(voxelCoords.z, voxelCoords.y, voxelCoords.x);
        }

        float vertexMinmaxAt(ivec3 voxelCoords)
        {            
            return insideVolume(voxelCoords) ? getA(voxelCoords.z, voxelCoords.y, voxelCoords.x) : 0.0;
        }

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        float computeVertexMinmax(ivec3 voxelCoords)
        {
            float v111 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset( 0,  0,  0, axis, octant)}));
            float v110 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset( 0,  0, -1, axis, octant)}));
            float v100 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset( 0, -1, -1, axis, octant)}));
            float v010 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset(-1,  0, -1, axis, octant)}));
            float v000 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset(-1, -1, -1, axis, octant)}));

            float bottleneck = min4(v000, v010, v100, v110);

            return max(v111, bottleneck);
        }

        void main()
        {
            setOutput(computeVertexMinmax(outputCoords()));
        }
        `
    }
}

/**
 * One dynamic-programming propagation step for a single sweep slice.
 *
 * A is the current raw slice. B is the already-propagated previous slice in
 * the requested direction. The shader keeps the current vertex value unless
 * every incoming route has already seen a larger guaranteed bottleneck.
 */
class PropagateVertexMinmaxProgram implements GPGPUProgram
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
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool insideSlice(ivec3 sliceCoords)
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
            return insideSlice(sliceCoords) ? getB(sliceCoords.z, sliceCoords.y, sliceCoords.x) : 0.0;
        }

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        float computeVertexMinmax(ivec3 sliceCoords)
        {
            float v111 =  currentSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  0,  0, axis, octant)}));
            float v110 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  0, -1, axis, octant)}));
            float v100 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0, -1, -1, axis, octant)}));
            float v010 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1,  0, -1, axis, octant)}));
            float v000 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1, -1, -1, axis, octant)}));

            float bottleneck = min4(v000, v010, v100, v110);

            return max(v111, bottleneck);
        }

        void main()
        {
            setOutput(computeVertexMinmax(outputCoords()));
        }
        `
    }
}

/**
 * Classifies vertices for one oriented ray class.
 *
 * A contains the original vertex values. B contains propagated minmax values.
 * A vertex is shadowed when its value is within tolerance of the bottleneck
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
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool insideVolume(ivec3 voxelCoords)
        {
            return 
                voxelCoords.x >= minCoords.x && voxelCoords.x <= maxCoords.x &&
                voxelCoords.y >= minCoords.y && voxelCoords.y <= maxCoords.y &&
                voxelCoords.z >= minCoords.z && voxelCoords.z <= maxCoords.z;
        }

        ivec3 outputCoords()
        {
            ivec3 voxelCoords = getOutputCoords();
            return ivec3(voxelCoords.z, voxelCoords.y, voxelCoords.x);
        }

        float vertexValueAt(ivec3 voxelCoords)
        {
            return getA(voxelCoords.z, voxelCoords.y, voxelCoords.x);
        }

        float vertexMinmaxAt(ivec3 voxelCoords)
        {
            return insideVolume(voxelCoords) ? getB(voxelCoords.z, voxelCoords.y, voxelCoords.x) : 0.0;
        }

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        bool computeVertexShadow(ivec3 voxelCoords)
        {
            float v111 =  vertexValueAt(voxelCoords + ivec3(${voxelOffset( 0,  0,  0, axis, octant)}));
            float v110 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset( 0,  0, -1, axis, octant)}));
            float v100 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset( 0, -1, -1, axis, octant)}));
            float v010 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset(-1,  0, -1, axis, octant)}));
            float v000 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset(-1, -1, -1, axis, octant)}));

            float bottleneck = min4(v000, v010, v100, v110);
            float margin = v111 - bottleneck;

            return (margin < tolerance);
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

        this.outputShape = [depth - 1, height - 1, width - 1]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        ivec3 outputCoords()
        {
            ivec3 voxelCoords = getOutputCoords();
            return ivec3(voxelCoords.z, voxelCoords.y, voxelCoords.x);
        }

        bool cellShadowAt(ivec3 cellCoords)
        {
            cellCoords = clamp(cellCoords, minCoords, maxCoords);
            return getA(cellCoords.z, cellCoords.y, cellCoords.x) > 0.5;
        }

        bool computeVertexHole(ivec3 voxelCoords)
        {
            return cellShadowAt(voxelCoords + ivec3(${cellOffset(1, 1, 1, axis, octant)}));
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

        this.outputShape = [depth + 1, height + 1, width + 1]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});
            
        ivec3 outputCoords()
        {
            ivec3 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        bool vertexShadowAt(ivec3 voxelCoords)
        {
            voxelCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(voxelCoords.z, voxelCoords.y, voxelCoords.x) > 0.5;
        }

        bool computeCellShadow(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            return (
                vertexShadowAt(voxelCoords + ivec3(${cellOffset(1, 1, 1, axis, octant)})) &&
                vertexShadowAt(voxelCoords + ivec3(${cellOffset(1, 0, 1, axis, octant)})) &&
                vertexShadowAt(voxelCoords + ivec3(${cellOffset(0, 1, 1, axis, octant)})) &&
                vertexShadowAt(voxelCoords + ivec3(${cellOffset(0, 0, 1, axis, octant)})) &&
                vertexShadowAt(voxelCoords + ivec3(${cellOffset(1, 1, 0, axis, octant)})) &&
                vertexShadowAt(voxelCoords + ivec3(${cellOffset(1, 0, 0, axis, octant)})) &&
                vertexShadowAt(voxelCoords + ivec3(${cellOffset(0, 1, 0, axis, octant)})) 
            );
        }

        void main()
        {
            setOutput(float(computeCellShadow(outputCoords())));
        }
        `
    }
}

/**
 * Slow reference implementation of directional minmax propagation.
 */
async function iterateVertexMinmax(
    volume: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    verbose: boolean = false
): Promise<tf.Tensor3D>
{
    const dimension = su.axisToDimension(axis)
    const length = volume.shape[dimension]
    
    const shape = volume.shape as [number, number, number]
    const propagate = new IterateVertexMinmaxProgram(shape, axis, octant)

    let prev = volume.clone() 
    for (let i = 1; i !== length; i += 1)
    {
        const next = runWebGLProgram(propagate, [prev], 'float32', [], true)
        tf.dispose(prev)
        prev = next  as tf.Tensor3D

        await tf.nextFrame()
    }

    const vertexMinmax = prev
    if (verbose) logMean3d('vertexMinmax', vertexMinmax)

    return vertexMinmax as tf.Tensor3D
}

/**
 * Propagates directional minmax values one slice at a time along the selected
 * axis and octant. This is the fast path used by the public shadow-map helpers.
 */
function propagateVertexMinmax(
    volume: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    verbose: boolean = false
): tf.Tensor3D
{
    const dimension = su.axisToDimension(axis)
    const sign = su.getOctantSign(octant, dimension)
    
    const slices = unstack3d(volume, dimension)
    const shape = slices[0].shape as [number, number, number]
    const propagate = new PropagateVertexMinmaxProgram(shape, axis, octant)

    const backwards = sign === '-'
    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length // 0 : slices.length - 1
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step)
    {
        const next = runWebGLProgram(propagate, [slices[i], slices[i-step]], 'float32', [[i]], true)
        tf.dispose(slices[i])
        slices[i] = next
    }

    const vertexMinmax = stack3d(slices, dimension) 
    tf.dispose(slices)
    if (verbose) logMean3d('vertexMinmax', vertexMinmax)

    return vertexMinmax as tf.Tensor3D
}

/**
 * Converts propagated minmax values into a binary vertex shadow mask.
 */
function computeVertexShadows(
    vertexValues: tf.Tensor3D,
    vertexMinmax: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const program = new ComputeVertexShadowsProgram(vertexValues.shape, axis, octant)
    const vertexShadows = runWebGLProgram(program, [vertexValues, vertexMinmax], 'bool', [[tolerance]], true) 
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
    axis: su.Axis,
    octant: su.Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const vertexMinmax = propagateVertexMinmax(volume, axis, octant, verbose)
    const vertexShadows = computeVertexShadows(volume, vertexMinmax, axis, octant, tolerance, verbose)
    tf.dispose(vertexMinmax)

    const cellShadows = computeCellShadows(vertexShadows, axis, octant, verbose)
    tf.dispose(vertexShadows)

    return cellShadows
}

/**
 * Computes the conservative mask for both directions of the same ray family.
 */
export function computeBidirectionalShadowMap(
    volume: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    // Forward 
    const forwardOctant = octant
    const forwardVertexMinmax = propagateVertexMinmax(volume, axis, forwardOctant, verbose)
    const forwardVertexShadows = computeVertexShadows(volume, forwardVertexMinmax, axis, forwardOctant, tolerance, verbose)
    tf.dispose(forwardVertexMinmax)

    const forwardCellShadows = computeCellShadows(forwardVertexShadows, axis, forwardOctant, false)
    tf.dispose(forwardVertexShadows)
    if (verbose) logMean3d('forwardCellShadows', forwardCellShadows)

    // Backward 
    const backwardOctant = su.reverseOctant(forwardOctant)
    const backwardVertexHoles = computeVertexHoles(forwardCellShadows, axis, backwardOctant, verbose)
    const backwardVertexValues = tf.where(backwardVertexHoles, 0, volume) 
    tf.dispose(backwardVertexHoles)

    const backwardVertexMinmax = propagateVertexMinmax(backwardVertexValues, axis, backwardOctant, verbose)
    const backwardVertexShadows = computeVertexShadows(backwardVertexValues, backwardVertexMinmax, axis, backwardOctant, tolerance, verbose)
    tf.dispose([backwardVertexValues, backwardVertexMinmax])

    const backwardCellShadows = computeCellShadows(backwardVertexShadows, axis, backwardOctant, false)
    tf.dispose(backwardVertexShadows)
    if (verbose) logMean3d('backwardCellShadows', backwardCellShadows)

    // Bidirectional 
    const bidirectionalCellShadows = tf.logicalOr(forwardCellShadows, backwardCellShadows)
    tf.dispose([forwardCellShadows, backwardCellShadows])
    if (verbose) logMean3d('bidirectionalCellShadows', bidirectionalCellShadows)

    return bidirectionalCellShadows as tf.Tensor3D
}

/**
 * Computes a bidirectional mask and min-pools it into traversal blocks.
 *
 * Min pooling keeps a block rejected only when every covered cell is rejected.
 */
export function computeBidirectionalBlockShadowMap2(
    volume: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shadows = computeBidirectionalShadowMap(volume, axis, octant, tolerance, verbose)
    if (blockSize === 1) return shadows
    const blockShadows = minPool3d(shadows, blockSize, blockSize, 'same') 
    tf.dispose(shadows)
    if (verbose) logMean3d('blockShadows', blockShadows)
 
    return blockShadows as tf.Tensor3D
}

/**
 * Alternative block-first variant. It pools the input value bounds first, then
 * computes the bidirectional mask at block resolution.
 */
export function computeBidirectionalBlockShadowMap(
    volume: tf.Tensor3D,
    axis: su.Axis,
    octant: su.Octant,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    // Reduce 
    const minVertexValues = minPool3d(volume, blockSize, blockSize, 'valid')
    const maxVertexValues = maxPool3d(volume, blockSize, blockSize, 'valid')

    // Forward
    const forwardOctant = octant
    const forwardMinVertexValues = minVertexValues
    const forwardMaxVertexValues = maxVertexValues

    const forwardVertexMinmax = propagateVertexMinmax(forwardMinVertexValues, axis, forwardOctant, verbose)
    const forwardVertexShadows = computeVertexShadows(forwardMaxVertexValues, forwardVertexMinmax, axis, forwardOctant, tolerance, verbose)
    tf.dispose(forwardVertexMinmax)

    const forwardCellShadows = computeCellShadows(forwardVertexShadows, axis, forwardOctant, false)
    tf.dispose(forwardVertexShadows)
    if (verbose) logMean3d('forwardCellShadows', forwardCellShadows)

    // Backward 
    const backwardOctant = su.reverseOctant(forwardOctant)
    const backwardVertexHoles = computeVertexHoles(forwardCellShadows, axis, backwardOctant, verbose)
    const backwardMinVertexValues = tf.where(backwardVertexHoles, 0, minVertexValues) 
    const backwardMaxVertexValues = tf.where(backwardVertexHoles, 0, maxVertexValues) 
    tf.dispose([backwardVertexHoles, minVertexValues, maxVertexValues])

    const backwardVertexMinmax = propagateVertexMinmax(backwardMinVertexValues, axis, backwardOctant, verbose)
    tf.dispose(backwardMinVertexValues)

    const backwardVertexShadows = computeVertexShadows(backwardMaxVertexValues, backwardVertexMinmax, axis, backwardOctant, tolerance, verbose)
    tf.dispose([backwardMaxVertexValues, backwardVertexMinmax])

    const backwardCellShadows = computeCellShadows(backwardVertexShadows, axis, backwardOctant, false)
    tf.dispose(backwardVertexShadows)
    if (verbose) logMean3d('backwardCellShadows', backwardCellShadows)

    // Bidirectional 
    const bidirectionalCellShadows = tf.logicalOr(forwardCellShadows, backwardCellShadows)
    tf.dispose([forwardCellShadows, backwardCellShadows])
    if (verbose) logMean3d('bidirectionalCellShadows', bidirectionalCellShadows)

    return bidirectionalCellShadows as tf.Tensor3D
}
