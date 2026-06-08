
import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { unstack3dPacked } from './unstack3dPacked'
import { stack3dPacked } from './stack3dPacked'
import { minPool3dPacked, maxPool3dPacked } from './pool3dPacked'
import { where3dPacked } from './where3dPacked'
import * as su from '../../Utils/ShadowMapUtils'

/**
 * Packed directional shadow-map preprocessing for MIP distance maps.
 *
 * This is the vec4-lane version of GPGPUShadowMapPaths.ts. Instead of computing
 * one octant at a time, each tensor element stores four related ray classes in
 * RGBA lanes. The `sign` argument selects the dominant-axis direction; the lanes
 * carry the four transverse octants for that direction.
 *
 * TensorFlow tensors are shaped/indexed as [z, y, x], while the GLSL snippets
 * use ivec3(x, y, z). Offset helpers therefore work in z/y/x internally, then
 * emit x/y/z GLSL constructor arguments.
 *
 * Masks use 1 for rejected cells and 0 for cells that may still contribute to
 * the MIP. Input volumes are expected to be normalized and non-negative.
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
 * Reversed axes flip offset signs because voxel addresses move across the grid.
 */
function voxelOffset(
    x: number,
    y: number,
    z: number,
    axis: su.Axis,
    sign: su.Sign,
): string
{
    const octant = `${sign}${sign}${sign}` as su.Octant

    const { permute, reverse } = su.dominantAxisOctantToPermuteReverse(axis, octant)

    const zyx = su.applyPermutation([z, y, x], permute)

    for (const dimension of reverse) zyx[dimension] = -zyx[dimension]

    return xyz(zyx)
}

/**
 * Converts a canonical propagation offset for already-unstacked slices. Movement
 * along the sweep axis is removed because the previous sweep slice is supplied
 * as a separate input tensor.
 */
function sliceOffset(
    x: number,
    y: number,
    z: number,
    axis: su.Axis,
    sign: su.Sign,
): string
{
    const octant = `${sign}${sign}${sign}` as su.Octant

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
    sign: su.Sign,
): string
{
    const octant = `${sign}${sign}${sign}` as su.Octant

    const { permute, reverse } = su.dominantAxisOctantToPermuteReverse(axis, octant)

    const zyx = su.applyPermutation([z, y, x], permute)

    for (const dimension of reverse) zyx[dimension] = 1 - zyx[dimension]

    return xyz(zyx)
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

/**
 * Expands a scalar volume into packed vertex values. Each RGBA lane starts with
 * the same source value; later passes interpret lanes as different ray classes.
 */
class ComputeVertexValuesProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(shape: [number, number, number]) 
    {
        const [depth, height, width] = shape

        this.outputShape = [depth, height, width, 2, 2]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        ivec3 outputCoords()
        {
            ivec5 voxelCoords = getOutputCoords();
            return ivec3(voxelCoords.z, voxelCoords.y, voxelCoords.x);
        }

        float vertexValueAt(ivec3 voxelCoords)
        {
            return getA(voxelCoords.z, voxelCoords.y, voxelCoords.x);
        }

        void main()
        {
            setOutput(vec4(vertexValueAt(outputCoords())));
        }
        `
    }
}

/**
 * One dynamic-programming propagation step for a single sweep slice.
 *
 * A is the current raw packed slice. B is the already-propagated previous slice.
 * The RGBA lanes propagate four transverse octants at once, so the shader reads
 * a 3x3 neighborhood from B and forms one bottleneck per lane.
 */
class PropagateVertexMinmaxProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'slice', type: 'int' as const }]

    constructor(
        sliceShape: [number, number, number, 2, 2],
        axis: su.Axis = 'z',
        sign: su.Sign = '+',
    ) {
        const [depth, height, width, ] = sliceShape
        
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
            ivec5 sliceCoords = getOutputCoords();
            return ivec3(sliceCoords.z, sliceCoords.y, sliceCoords.x);
        }

        vec4 currentSliceAt(ivec3 sliceCoords)
        {
            return getA(sliceCoords.z, sliceCoords.y, sliceCoords.x, 0, 0);
        }

        vec4 previousSliceAt(ivec3 sliceCoords)
        {            
            return insideSlice(sliceCoords) ? getB(sliceCoords.z, sliceCoords.y, sliceCoords.x, 0, 0) : vec4(0.0);
        }

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        vec4 computeVertexMinmax(ivec3 sliceCoords)
        {
            vec4 v111 =  currentSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  0,  0, axis, sign)}));
            vec4 v000 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1, -1, -1, axis, sign)}));
            vec4 v100 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0, -1, -1, axis, sign)}));
            vec4 v200 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 1, -1, -1, axis, sign)}));
            vec4 v010 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1,  0, -1, axis, sign)}));
            vec4 v110 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  0, -1, axis, sign)}));
            vec4 v210 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 1,  0, -1, axis, sign)}));
            vec4 v020 = previousSliceAt(sliceCoords + ivec3(${sliceOffset(-1,  1, -1, axis, sign)}));
            vec4 v120 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 0,  1, -1, axis, sign)}));
            vec4 v220 = previousSliceAt(sliceCoords + ivec3(${sliceOffset( 1,  1, -1, axis, sign)}));

            vec4 bottlenecks = vec4(
                min4(v000.r, v010.r, v100.r, v110.r), // +++
                min4(v100.g, v110.g, v200.g, v210.g), // -++
                min4(v010.b, v020.b, v110.b, v120.b), // +-+
                min4(v110.a, v120.a, v210.a, v220.a)  // --+
            );
            
            vec4 vertexMinmax = max(v111, bottlenecks);
            
            return vertexMinmax;
        }

        void main()
        {
            setOutput(computeVertexMinmax(outputCoords()));
        }
        `
    }
}

/**
 * Classifies packed vertices for four ray classes at once.
 *
 * A contains original packed vertex values. B contains propagated minmax values.
 * A lane is shadowed when its value is within tolerance of the lane bottleneck.
 */
class ComputeVertexShadowsProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'tolerance', type: 'float' as const }]

    constructor(
        shape: [number, number, number, 2, 2],
        axis: su.Axis = 'z',
        sign: su.Sign = '+'
    ) {
        const [depth, height, width, ] = shape
    
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
            ivec5 voxelCoords = getOutputCoords();
            return ivec3(voxelCoords.z, voxelCoords.y, voxelCoords.x);
        }

        vec4 vertexValueAt(ivec3 voxelCoords)
        {
            return getA(voxelCoords.z, voxelCoords.y, voxelCoords.x, 0, 0);
        }

        vec4 vertexMinmaxAt(ivec3 voxelCoords)
        {
            return insideVolume(voxelCoords) ? getB(voxelCoords.z, voxelCoords.y, voxelCoords.x, 0, 0) : vec4(0.0);
        }

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        bvec4 computeVertexShadows(ivec3 voxelCoords)
        {
            vec4 v111 =  vertexValueAt(voxelCoords + ivec3(${voxelOffset( 0,  0,  0, axis, sign)}));
            vec4 v000 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset(-1, -1, -1, axis, sign)}));
            vec4 v100 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset( 0, -1, -1, axis, sign)}));
            vec4 v200 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset( 1, -1, -1, axis, sign)}));
            vec4 v010 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset(-1,  0, -1, axis, sign)}));
            vec4 v110 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset( 0,  0, -1, axis, sign)}));
            vec4 v210 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset( 1,  0, -1, axis, sign)}));
            vec4 v020 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset(-1,  1, -1, axis, sign)}));
            vec4 v120 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset( 0,  1, -1, axis, sign)}));
            vec4 v220 = vertexMinmaxAt(voxelCoords + ivec3(${voxelOffset( 1,  1, -1, axis, sign)}));
           
            vec4 bottlenecks = vec4(
                min4(v000.r, v010.r, v100.r, v110.r), // +++ 
                min4(v100.g, v110.g, v200.g, v210.g), // -++
                min4(v010.b, v020.b, v110.b, v120.b), // +-+
                min4(v110.a, v120.a, v210.a, v220.a)  // --+
            );

            bvec4 vertexShadows = lessThan(v111 - bottlenecks, vec4(tolerance));

            return vertexShadows;
        }

        void main()
        {
            setOutput(vec4(computeVertexShadows(outputCoords())));
        }
        `
    }
}

/**
 * Builds the reverse-pass hole tensor from a packed forward cell mask.
 *
 * During the backward sweep, cells already rejected by the forward sweep should
 * not act as solid occluders. The lane shuffle here maps each forward lane to
 * its reversed ray class.
 */
class ComputeVertexHolesProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        shape: [number, number, number, 2, 2],
        axis: su.Axis = 'z',
        sign: su.Sign = '+'
    ) {
        const [depth, height, width, ] = shape

        this.outputShape = [depth - 1, height - 1, width - 1, 2, 2]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        ivec3 outputCoords()
        {
            ivec5 voxelCoords = getOutputCoords();
            return ivec3(voxelCoords.z, voxelCoords.y, voxelCoords.x);
        }

        bvec4 cellShadowAt(ivec3 cellCoords)
        {
            cellCoords = clamp(cellCoords, minCoords, maxCoords);
            return greaterThan(getA(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0), vec4(0.5));
        }

        bvec4 computeVertexHoles(ivec3 voxelCoords)
        {
            bvec4 vertexHoles = bvec4(
                cellShadowAt(voxelCoords + ivec3(${cellOffset(1, 1, 1, axis, sign)})).r, // ---
                cellShadowAt(voxelCoords + ivec3(${cellOffset(0, 1, 1, axis, sign)})).g, // +--
                cellShadowAt(voxelCoords + ivec3(${cellOffset(1, 0, 1, axis, sign)})).b, // -+-
                cellShadowAt(voxelCoords + ivec3(${cellOffset(0, 0, 1, axis, sign)})).a  // ++-
            );
   
            return vertexHoles;
        }

        void main()
        {            
            setOutput(vec4(computeVertexHoles(outputCoords())));
        }
        `
    }
}

/**
 * Promotes packed vertex shadows to packed cell shadows.
 *
 * A lane is rejected only when all relevant corners for that ray class are
 * shadowed, keeping the final culling mask conservative.
 */
class ComputeCellShadowsProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        shape: [number, number, number, 2, 2],
        axis: su.Axis = 'z',
        sign: su.Sign = '+'
    ) {
        const [depth, height, width, ] = shape

        this.outputShape = [depth + 1, height + 1, width + 1, 2, 2]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});
            
        ivec3 outputCoords()
        {
            ivec5 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        bvec4 vertexShadowsAt(ivec3 voxelCoords)
        {
            voxelCoords = clamp(voxelCoords, minCoords, maxCoords);
            return greaterThan(getA(voxelCoords.z, voxelCoords.y, voxelCoords.x, 0, 0), vec4(0.5));
        }

        bvec4 computeCellShadows(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            bvec4 v111 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(1, 1, 1, axis, sign)}));
            bvec4 v110 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(1, 1, 0, axis, sign)}));
            bvec4 v101 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(1, 0, 1, axis, sign)}));
            bvec4 v011 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(0, 1, 1, axis, sign)}));
            bvec4 v100 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(1, 0, 0, axis, sign)}));
            bvec4 v010 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(0, 1, 0, axis, sign)}));
            bvec4 v001 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(0, 0, 1, axis, sign)}));
            bvec4 v000 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(0, 0, 0, axis, sign)}));

            bvec4 cellShadows = bvec4(
                v111.r && v101.r && v011.r && v001.r && v110.r && v100.r && v010.r, // +++
                v111.g && v101.g && v011.g && v001.g && v110.g && v000.g && v010.g, // -++
                v111.b && v101.b && v011.b && v001.b && v110.b && v100.b && v000.b, // +-+
                v111.a && v101.a && v011.a && v001.a && v000.a && v100.a && v010.a  // --+
            );

            return cellShadows;
        }

        void main()
        {
            setOutput(vec4(computeCellShadows(outputCoords())));
        }
        `
    }
}

/**
 * Converts scalar volume values into packed RGBA vertex values.
 */
function computeVertexValues(
    volume: tf.Tensor3D
): tf.Tensor5D
{
    const shape = volume.shape as [number, number, number]
    const program = new ComputeVertexValuesProgram(shape)
    const vertexValues = runWebGLProgram(program, [volume], 'float32', [], true) 
    return vertexValues as tf.Tensor5D
}

/**
 * Propagates packed directional minmax values one slice at a time along the
 * selected dominant axis and sign.
 */
function propagateVertexMinmax(
    vertexValues: tf.Tensor5D,
    axis: su.Axis,
    sign: su.Sign,
    verbose: boolean = false
): tf.Tensor5D
{
    const dimension = su.axisToDimension(axis)
    const backwards = sign === '-'
    
    const slices = unstack3dPacked(vertexValues, dimension)
    const shape = slices[0].shape as [number, number, number, 2, 2]
    const propagate = new PropagateVertexMinmaxProgram(shape, axis, sign)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? 0 : slices.length - 1
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step)
    {
        const next = runWebGLProgram(propagate, [slices[i], slices[i-step]], 'float32', [[i]], true)
        tf.dispose(slices[i])
        slices[i] = next
    }

    const vertexMinmax = stack3dPacked(slices, dimension) 
    tf.dispose(slices)
    if (verbose) logMean3d('vertexMinmax', vertexMinmax)

    return vertexMinmax as tf.Tensor5D
}

/**
 * Converts propagated packed minmax values into a packed vertex shadow mask.
 */
function computeVertexShadows(
    vertexValues: tf.Tensor5D,
    vertexMinmax: tf.Tensor5D,
    axis: su.Axis,
    sign: su.Sign,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = vertexValues.shape as [number, number, number, 2, 2]
    const program = new ComputeVertexShadowsProgram(shape, axis, sign)
    const vertexShadows = runWebGLProgram(program, [vertexValues, vertexMinmax], 'bool', [[tolerance]], true) 
    if (verbose) logMean3d('vertexShadows', vertexShadows)

    return vertexShadows as tf.Tensor5D
}

/**
 * Converts a packed vertex shadow mask into a packed cell shadow mask.
 */
function computeCellShadows(
    vertexShadows: tf.Tensor5D,
    axis: su.Axis,
    sign: su.Sign,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = vertexShadows.shape as [number, number, number, 2, 2]
    const program = new ComputeCellShadowsProgram(shape, axis, sign)
    const cellShadows = runWebGLProgram(program, [vertexShadows], 'bool', [], true) 
    if (verbose) logMean3d('cellShadows', cellShadows)

    return cellShadows as tf.Tensor5D
}

/**
 * Converts packed forward cell shadows into packed vertex holes for reverse propagation.
 */
function computeVertexHoles(
    cellShadows: tf.Tensor5D,
    axis: su.Axis,
    sign: su.Sign,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = cellShadows.shape as [number, number, number, 2, 2]
    const program = new ComputeVertexHolesProgram(shape, axis, sign)
    const vertexHoles = runWebGLProgram(program, [cellShadows], 'bool', [], true) 
    if (verbose) logMean3d('vertexHoles', vertexHoles)

    return vertexHoles as tf.Tensor5D
}

/**
 * Computes four unidirectional conservative cell-rejection masks in packed lanes.
 */
export function computeUnidirectionalShadowMap(
    volume: tf.Tensor3D,
    axis: su.Axis,
    sign: su.Sign,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor5D
{
    const vertexValues = computeVertexValues(volume)
    const vertexMinmax = propagateVertexMinmax(vertexValues, axis, sign, verbose)
    const vertexShadows = computeVertexShadows(vertexValues, vertexMinmax, axis, sign, tolerance, verbose)
    tf.dispose([vertexValues, vertexMinmax])

    const cellShadows = computeCellShadows(vertexShadows, axis, sign, verbose)
    tf.dispose(vertexShadows)

    return cellShadows as tf.Tensor5D
}

/**
 * Computes packed conservative masks for both directions of the same ray family.
 */
export function computeBidirectionalShadowMap(
    volume: tf.Tensor3D,
    axis: su.Axis,
    sign: su.Sign,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor5D
{
    const vertexValues = computeVertexValues(volume)

    // Forward
    const forwardSign = sign
    const forwardVertexValues = vertexValues
    
    const forwardVertexMinmax = propagateVertexMinmax(forwardVertexValues, axis, forwardSign, verbose)
    const forwardVertexShadows = computeVertexShadows(forwardVertexValues, forwardVertexMinmax, axis, forwardSign, tolerance, verbose)
    tf.dispose(forwardVertexMinmax)

    const forwardCellShadows = computeCellShadows(forwardVertexShadows, axis, forwardSign, false)
    tf.dispose(forwardVertexShadows)
    if (verbose) logMean3d('forwardCellShadows', forwardCellShadows)

    // Backwards
    const backwardSign = su.reverseSign(sign)
    const backwardVertexHoles = computeVertexHoles(forwardCellShadows, axis, backwardSign, verbose)
    const backwardVertexValues = where3dPacked(backwardVertexHoles, 0, vertexValues)
    tf.dispose([backwardVertexHoles, vertexValues])

    const backwardVertexMinmax = propagateVertexMinmax(backwardVertexValues, axis, backwardSign, verbose)
    const backwardVertexShadows = computeVertexShadows(backwardVertexValues, backwardVertexMinmax, axis, backwardSign, tolerance, verbose)
    tf.dispose(backwardVertexMinmax)

    const backwardCellShadows = computeCellShadows(backwardVertexShadows, axis, backwardSign, false)
    tf.dispose(backwardVertexShadows)
    if (verbose) logMean3d('backwardCellShadows', backwardCellShadows)

    // Bidirectional
    const bidirectionalCellShadows = tf.logicalOr(forwardCellShadows, backwardCellShadows)
    tf.dispose([forwardCellShadows, backwardCellShadows])
    if (verbose) logMean3d('bidirectionalCellShadows', bidirectionalCellShadows)

    return bidirectionalCellShadows as tf.Tensor5D
}

/**
 * Computes a packed bidirectional mask and min-pools it into traversal blocks.
 */
export function computeBidirectionalBlockShadowMap(
    volume: tf.Tensor3D,
    axis: su.Axis,
    sign: su.Sign,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor5D
{
    const shadows = computeBidirectionalShadowMap(volume, axis, sign, tolerance, verbose)
    if (blockSize === 1) return shadows
    const blockShadows = minPool3dPacked(shadows, blockSize, blockSize, 'same') 
    tf.dispose(shadows)
    if (verbose) logMean3d('blockShadows', blockShadows)

    return blockShadows
}

/**
 * Alternative block-first variant. It pools packed input value bounds first,
 * then computes the bidirectional mask at block resolution.
 */
export function computeBidirectionalBlockShadowMap2(
    volume: tf.Tensor3D,
    axis: su.Axis,
    sign: su.Sign,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor5D
{
    // Reduce
    const vertexValues = computeVertexValues(volume)
    const minVertexValues = minPool3dPacked(vertexValues, blockSize, blockSize, 'valid')
    const maxVertexValues = maxPool3dPacked(vertexValues, blockSize, blockSize, 'valid')
    tf.dispose(vertexValues)

    // Forward
    const forwardSign = sign
    const forwardMinVertexValues = minVertexValues
    const forwardMaxVertexValues = maxVertexValues
   
    const forwardVertexMinmax = propagateVertexMinmax(forwardMinVertexValues, axis, forwardSign, verbose)
    const forwardVertexShadows = computeVertexShadows(forwardMaxVertexValues, forwardVertexMinmax, axis, forwardSign, tolerance, verbose)
    tf.dispose(forwardVertexMinmax)

    const forwardCellShadows = computeCellShadows(forwardVertexShadows, axis, forwardSign, verbose)
    tf.dispose(forwardVertexShadows)

    // Backwards
    const backwardSign = su.reverseSign(sign)
    const backwardVertexHoles = computeVertexHoles(forwardCellShadows, axis, backwardSign, verbose)
    const backwardMinVertexValues = where3dPacked(backwardVertexHoles, 0, minVertexValues)
    tf.dispose(minVertexValues)
    const backwardMaxVertexValues = where3dPacked(backwardVertexHoles, 0, maxVertexValues)
    tf.dispose([maxVertexValues, backwardVertexHoles])

    const backwardVertexMinmax = propagateVertexMinmax(backwardMinVertexValues, axis, backwardSign, verbose)
    tf.dispose(backwardMinVertexValues)

    const backwardVertexShadows = computeVertexShadows(backwardMaxVertexValues, backwardVertexMinmax, axis, backwardSign, tolerance, verbose)
    tf.dispose([backwardMaxVertexValues, backwardVertexMinmax])

    const backwardCellShadows = computeCellShadows(backwardVertexShadows, axis, backwardSign, verbose)
    tf.dispose(backwardVertexShadows)

    // Bidirectional
    const bidirectionalCellShadows = tf.logicalOr(forwardCellShadows, backwardCellShadows)
    tf.dispose([forwardCellShadows, backwardCellShadows])
    if (verbose) logMean3d('bidirectionalCellShadows', bidirectionalCellShadows)

    return bidirectionalCellShadows as tf.Tensor5D
}

