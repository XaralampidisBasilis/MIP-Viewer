/**
 * Extended anisotropic shadow maps for maximum intensity projection.
 *
 * This is the difference-domain version of the cell-based shadow-map idea from
 * Mroz, Hauser, and Groeller, "Interactive High-Quality Maximum Intensity
 * Projection", section 2, especially the section 2.2 preprocessing/visibility
 * test for trilinearly interpolated cells.
 *
 * The algorithm is written around one canonical cell orientation: rays
 * propagate in local +z, so z is the dominant axis in the GLSL stencil. Any
 * requested dominant-axis/octant pair is converted into that canonical case by
 * permuting and reversing offsets before they are injected into the shaders.
 *
 * Each cell is a unit cube with eight voxel corner values. The corner suffix
 * is the local (x, y, z) coordinate used by trilinear interpolation, so c000 is
 * (0, 0, 0), c100 is (1, 0, 0), ..., and c111 is (1, 1, 1).
 *
 * In the canonical initial-margin pass we compare the target corner c111 with
 * the four corners on the previous z face: c000, c010, c100, and c110. The
 * packed lanes store previous-face-minus-c111 margins in that order. Positive
 * means a previous cell corner can dominate c111 in a maximum-intensity
 * projection.
 *
 * Section 2.2 of the paper motivates this as a conservative cell rejection
 * test: if the maximum a cell can contribute is already bounded by values seen
 * before the ray reaches it, that cell cannot change the final MIP value.
 * Instead of storing explicit face minima/maxima, this implementation stores
 * differences and propagates the guaranteed positive margin through monotone
 * ray classes.
 *
 * The exported API keeps the historical "ShadowMap" names used elsewhere in
 * the viewer. Internally, the final shadow-map values are described as cell
 * masks: 1 means the cell is rejected for a ray class, 0 means it can still
 * contribute to the MIP.
 */
import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { unstack3dPacked } from './unstack_packed_keepDims_webgl'
import { stack3dPacked } from './stack_packed_keepDims_webgl'
import {
    type Axis,
    type Sign,
    axisToDimension,
    setOctantSign,
    reverseOctant,
    applyPermutation,
    dominantAxisOctantToPermuteReverse,
} from '../../Utils/ShadowMapUtils'
import { minPool3d } from './pool3d'
import { compute } from 'three/webgpu'

type Shape3 = [number, number, number]
type Shape3Packed = [number, number, number, 2, 2]
type CoordExpr = number | string

/**
 * Convert a canonical sweep-space offset into the physical tensor orientation.
 * The TypeScript side thinks in local x/y/z offsets; TensorFlow stores tensors
 * as z/y/x, so applyPermutation works in z/y/x order and xyz emits GLSL ivec3
 * arguments.
 */
function xyz(zyx: [CoordExpr, CoordExpr, CoordExpr]): string
{
    return [zyx[2], zyx[1], zyx[0]].join(', ')
}

/**
 * Offset for voxel-vertex addressing. Reversal flips the sign of the offset, 
 * over the dominant axis.
 */
function voxelOffset(
    x: number,
    y: number,
    z: number,
    axis: Axis,
    sign: Sign,
): string
{
    const octant = setOctantSign('+++', axis, sign)

    const { permute, reverse } = dominantAxisOctantToPermuteReverse(axis, octant)

    const zyx = applyPermutation([z, y, x], permute)

    for (const dimension of reverse) zyx[dimension] = -zyx[dimension]

    return xyz(zyx)
}

/**
 * Offset for slice propagation. The propagated neighbor is supplied as a
 * separate 2D slice tensor, so any movement along the sweep axis must be
 * erased after orientation is applied.
 */
function sliceOffset(
    x: number,
    y: number,
    z: number,
    axis: Axis,
    sign: Sign,
): string
{
    const octant = setOctantSign('+++', axis, sign)

    const { permute, reverse } = dominantAxisOctantToPermuteReverse(axis, octant)

    const zyx = applyPermutation([z, y, x], permute)

    for (const dimension of reverse) zyx[dimension] = -zyx[dimension]

    zyx[permute[0]] = 0

    return xyz(zyx)
}

/**
 * Offset for cell-corner addressing. Reversal mirrors a unit cell corner, so
 * coordinate 0 becomes 1 and coordinate 1 becomes 0 instead of simply flipping
 * the sign.
 */
function cellOffset(
    x: number,
    y: number,
    z: number,
    axis: Axis,
    sign: Sign,
): string
{
    const octant = setOctantSign('+++', axis, sign)

    const { permute, reverse } = dominantAxisOctantToPermuteReverse(axis, octant)

    const zyx = applyPermutation([z, y, x], permute)

    for (const dimension of reverse) zyx[dimension] = 1 - zyx[dimension]

    return xyz(zyx)
}

class ComputeVertexValuesProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: Shape3Packed
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(shape: Shape3) 
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

        float voxelValueAt(ivec3 voxelCoords)
        {
            return getA(voxelCoords.z, voxelCoords.y, voxelCoords.x);
        }

        void main()
        {
            setOutput(vec4(voxelValueAt(outputCoords())));
        }
        `
    }
}

/**
 * Dynamic-programming propagation pass.
 *
 * A and B are neighboring slices of the same margin volume. A provides the
 * current slice local margins, while B provides the already-propagated previous
 * slice. The minimum lane in each incoming cell is the limiting margin through
 * that route.
 *
 * Mathematically, for each incoming offset r:
 *
 *     Delta_r(i) += max(min_s Delta_s(i-r), 0)
 *
 * The min over lanes makes the guarantee conservative over all monotone paths
 * that can enter the cell. The max with zero keeps only the guaranteed
 * non-negative part of the already-seen MIP prefix.
 */
class PropagateVertexMinmaxProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: Shape3Packed
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'slice', type: 'int' as const }]

    constructor(
        sliceShape: Shape3Packed,
        axis: Axis = 'z',
        sign: Sign = '+',
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
            return ivec3(sliceCoords.z, sliceCoords.y, sliceCoords.x, 0, 0);
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
                min4(v010.g, v020.g, v110.g, v120.g), // +-+
                min4(v100.b, v110.b, v200.b, v210.b), // -++
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
 * Builds the reverse-pass gate tensor from the forward cell mask.
 *
 * Each lane is 1 when the corresponding forward corner is not rejected. The
 * reverse propagation shader reads this as "the reverse rejection is allowed
 * to grow through this lane".
 *
 * In the paper terminology, this is the "hollow" handling for the second pass:
 * cells rejected by the first directional test should not act as solid
 * occluders for the opposite direction.
 */
class ComputeVertexShadowsProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: Shape3Packed
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'tolerance', type: 'float' as const }]

    constructor(
        shape: Shape3Packed,
        axis: Axis = 'z',
        sign: Sign = '+'
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
            vec4 v111 =  currentSliceAt(sliceCoords + ivec3(${voxelOffset( 0,  0,  0, axis, sign)}));
            vec4 v000 = previousSliceAt(sliceCoords + ivec3(${voxelOffset(-1, -1, -1, axis, sign)}));
            vec4 v100 = previousSliceAt(sliceCoords + ivec3(${voxelOffset( 0, -1, -1, axis, sign)}));
            vec4 v200 = previousSliceAt(sliceCoords + ivec3(${voxelOffset( 1, -1, -1, axis, sign)}));
            vec4 v010 = previousSliceAt(sliceCoords + ivec3(${voxelOffset(-1,  0, -1, axis, sign)}));
            vec4 v110 = previousSliceAt(sliceCoords + ivec3(${voxelOffset( 0,  0, -1, axis, sign)}));
            vec4 v210 = previousSliceAt(sliceCoords + ivec3(${voxelOffset( 1,  0, -1, axis, sign)}));
            vec4 v020 = previousSliceAt(sliceCoords + ivec3(${voxelOffset(-1,  1, -1, axis, sign)}));
            vec4 v120 = previousSliceAt(sliceCoords + ivec3(${voxelOffset( 0,  1, -1, axis, sign)}));
            vec4 v220 = previousSliceAt(sliceCoords + ivec3(${voxelOffset( 1,  1, -1, axis, sign)}));
           
            vec4 bottlenecks = vec4(
                min4(v000.r, v010.r, v100.r, v110.r), // +++ 
                min4(v010.g, v020.g, v110.g, v120.g), // +-+
                min4(v100.b, v110.b, v200.b, v210.b), // -++
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

class ComputeVertexHolesProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: Shape3Packed
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        shape: Shape3Packed,
        axis: Axis = 'z',
        sign: Sign = '+'
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
                cellShadowAt(voxelCoords + ivec3(${cellOffset(1, 1, 1, axis, sign)})), // +++
                cellShadowAt(voxelCoords + ivec3(${cellOffset(1, 0, 1, axis, sign)})), // +-+
                cellShadowAt(voxelCoords + ivec3(${cellOffset(0, 1, 1, axis, sign)})), // -++
                cellShadowAt(voxelCoords + ivec3(${cellOffset(0, 0, 1, axis, sign)}))  // --+
            );
   
            return vertexHoles;
   
        }

        void main()
        {            
            setOutput(float(computeVertexHoles(outputCoords())));
        }
        `
    }
}

/**
 * Converts propagated vertex margins into a binary cell mask.
 *
 * The margin tensor stores vec4 margins to the four relevant cell vertices.
 * The output mask stores one binary value per cell-mask entry: 1 means that
 * the cell is conservatively rejected for this ray class.
 *
 * This is the conservative rejection predicate: every trilinear corner sample
 * the cell can expose to the ray class is already dominated, up to tolerance.
 */
class ComputeCellShadowsProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: Shape3Packed
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        shape: Shape3Packed,
        axis: Axis = 'z',
        sign: Sign = '+'
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

            bvec4 v111 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(1, 1, 1, axis, sign)}))
            bvec4 v110 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(1, 1, 0, axis, sign)}))
            bvec4 v101 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(1, 0, 1, axis, sign)}))
            bvec4 v011 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(0, 1, 1, axis, sign)}))
            bvec4 v100 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(1, 0, 0, axis, sign)}))
            bvec4 v010 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(0, 1, 0, axis, sign)}))
            bvec4 v001 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(0, 0, 1, axis, sign)}))
            bvec4 v000 = vertexShadowsAt(voxelCoords + ivec3(${cellOffset(0, 0, 0, axis, sign)}))

            bvec4 cellShadows = bvec4(
                v111.r && v101.r && v011.r && v001.r && v110.r && v100.r && v010.r, // +++
                v111.g && v101.g && v011.g && v001.g && v110.g && v100.g && v000.g, // +-+
                v111.b && v101.b && v011.b && v001.b && v110.b && v000.b && v010.b, // -++
                v111.a && v101.a && v011.a && v001.a && v000.a && v100.a && v010.a  // --+
            );

            return cellShadows;
        }

        void main()
        {
            setOutput(float(computeCellShadows(outputCoords())));
        }
        `
    }
}


export function computeVertexValues(volume: tf.Tensor3D): tf.Tensor5D
{
    const shape = volume.shape as Shape3
    const program = new ComputeVertexValuesProgram(shape)
    const vertexValues = runWebGLProgram(program, [volume], 'float32', [], true) 
    return vertexValues as tf.Tensor5D
}

/**
 * Runs the initial-margin pass and propagates margins through the volume one slice at a
 * time along the selected sweep axis.
 *
 * Slicing is done on the CPU side because each propagated slice depends on the
 * previous propagated slice. Each slice update is still a full WebGL pass.
 *
 * Conceptually this fills the directional table for one ray class from section
 * 2.2: every entry says whether the corresponding trilinear cell can be
 * skipped for rays in that class.
 */
export function computeVertexMinmax(
    vertexValues: tf.Tensor5D,
    axis: Axis,
    sign: Sign,
    verbose: boolean = false
): tf.Tensor5D
{
    const dimension = axisToDimension(axis)
    const backwards = sign === '-'
    
    const slices = unstack3dPacked(vertexValues, dimension)
    const shape = slices[0].shape as Shape3Packed
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
    if (verbose) logMean('vertexMinmax', vertexMinmax)

    return vertexMinmax as tf.Tensor5D
}

/**
 * Converts propagated vertex margins into a binary 3D cell mask.
 */
export function computeVertexShadows(
    vertexValues: tf.Tensor5D,
    vertexMinmax: tf.Tensor5D,
    axis: Axis,
    sign: Sign,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = vertexValues.shape as Shape3Packed
    const program = new ComputeVertexShadowsProgram(shape, axis, sign)
    const vertexShadows = runWebGLProgram(program, [vertexValues, vertexMinmax], 'bool', [[tolerance]], true) 
    if (verbose) logMean('vertexShadows', vertexShadows)

    return vertexShadows as tf.Tensor5D
}

/**
 * Converts propagated vertex margins into a binary 3D cell mask.
 */
export function computeCellShadows(
    vertexShadows: tf.Tensor5D,
    axis: Axis,
    sign: Sign,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = vertexShadows.shape as Shape3Packed
    const program = new ComputeCellShadowsProgram(shape, axis, sign)
    const cellShadows = runWebGLProgram(program, [vertexShadows], 'bool', [], true) 
    if (verbose) logMean('cellShadows', cellShadows)

    return cellShadows as tf.Tensor5D
}

export function computeVertexHoles(
    cellShadows: tf.Tensor5D,
    axis: Axis,
    sign: Sign,
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = cellShadows.shape as Shape3Packed
    const program = new ComputeVertexHolesProgram(shape, axis, sign)
    const vertexHoles = runWebGLProgram(program, [cellShadows], 'bool', [], true) 
    if (verbose) logMean('vertexHoles', vertexHoles)

    return vertexHoles as tf.Tensor5D
}

/**
 * Computes one directed cell mask for one dominant axis and octant.
 *
 * This is the core public operation: initialize local margins, propagate them
 * along the selected direction, then classify rejected cells.
 */
export function computeUnidirectionalShadowMap(
    volume: tf.Tensor3D,
    axis: Axis,
    sign: Sign,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor5D
{
    const vertexValues = computeVertexValues(volume)
    const vertexMinmax = computeVertexMinmax(vertexValues, axis, sign, verbose)
    const vertexShadows = computeVertexShadows(vertexValues, vertexMinmax, axis, sign, tolerance, verbose)
    tf.dispose([vertexValues, vertexMinmax])

    const cellShadows = computeCellShadows(vertexShadows, axis, sign, verbose)
    tf.dispose(vertexShadows)

    return cellShadows as tf.Tensor5D
}

/**
 * Computes a cell mask that considers both directions along the same oriented
 * line family.
 *
 * The forward pass is computed normally. The backward pass uses the forward
 * cell mask as a gate so the reverse propagation is consistent with cells
 * already rejected by the forward sweep. The two binary masks are then OR-ed
 * together.
 */
export function computeBidirectionalShadowMap(
    volume: tf.Tensor3D,
    axis: Axis,
    sign: Sign,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const backwardOctant = reverseOctant(octant)
    const forwardShadows = computeUnidirectionalShadowMap(volume, axis, sign, tolerance, verbose)

    const vertexHoles = computeVertexHoles(forwardShadows, axis, backwardOctant, verbose)
    const vertexValues = tf.where(vertexHoles, 0, volume) 
    tf.dispose(vertexHoles)

    const backwardShadows = computeUnidirectionalShadowMap(vertexValues, axis, backwardOctant, tolerance, verbose)
    tf.dispose(vertexValues)

    const shadows = tf.logicalOr(forwardShadows, backwardShadows)
    tf.dispose([forwardShadows, backwardShadows])
    if (verbose) logMean('shadows', shadows)

    return shadows as tf.Tensor3D
}

/**
 * Computes a bidirectional cell mask and reduces it into blocks. Binary min
 * pooling marks a block rejected only if every covered cell-mask entry is
 * rejected, preserving the conservative guarantee of the section 2.2 culling
 * test at a coarser traversal level.
 */
export function computeBidirectionalBlockShadowMap(
    volume: tf.Tensor3D,
    axis: Axis,
    sign: Sign,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shadows = computeBidirectionalShadowMap(volume, axis, octant, tolerance, verbose)
    if (blockSize === 1) return shadows

    const blockShadows = minPool3d(shadows, blockSize, blockSize, 'same') as tf.Tensor3D
    tf.dispose(shadows)
    if (verbose) logMean('blockShadows', blockShadows)

    return blockShadows
}

/**
 * Logs the mean value of a 3D/packed tensor over spatial axes. This is a cheap
 * way to inspect how much of a mask is active without downloading the whole
 * texture.
 */
function logMean(name: string, tensor: tf.Tensor): void
{
    console.log(name, tf.tidy(() => tensor.mean([0, 1, 2]).dataSync()))
}

/**
 * Thin wrapper around TensorFlow.js WebGL program execution that returns a
 * normal Tensor object from the backend TensorInfo.
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
