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
import { stack3dPacked } from './stack_packed_keepDims_webgl'
import { unstack3dPacked } from './unstack_packed_keepDims_webgl'
import {
    type Axis,
    type Octant,
    type Permute,
    type Reverse,
    applyPermutation,
    complementReverse,
    inversePermutation,
    dominantAxisOctantToPermuteReverse,
    reverseOctant,
} from '../../Utils/ShadowMapUtils'
import { minPool3d } from './pool3d'

type Shape3 = [number, number, number]
type PackedShape3 = [number, number, number, 2, 2]
type CoordExpr = number | string

/**
 * Convert a canonical sweep-space offset into the physical tensor orientation.
 * The TypeScript side thinks in local x/y/z offsets; TensorFlow stores tensors
 * as z/y/x, so applyPermutation works in z/y/x order and xyz emits GLSL ivec3
 * arguments.
 */
function xyz(coordsZyx: [CoordExpr, CoordExpr, CoordExpr]): string
{
    return [coordsZyx[2], coordsZyx[1], coordsZyx[0]].join(', ')
}

/**
 * Offset for voxel-vertex addressing. Reversal flips the sign of the offset, 
 * over the dominant axis.
 */
function voxelOffset(
    x: number,
    y: number,
    z: number,
    permute: Permute,
    reverse: Reverse
): string
{
    const o = applyPermutation([z, y, x], permute)

    for (const axis of reverse) o[axis] = -o[axis]

    return xyz(o)
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
    permute: Permute,
    reverse: Reverse
): string
{
    const o = applyPermutation([z, y, x], permute)

    for (const axis of reverse) o[axis] = -o[axis]

    o[permute[0]] = 0

    return xyz(o)
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
    permute: Permute,
    reverse: Reverse
): string
{
    const o = applyPermutation([z, y, x], permute)

    for (const axis of reverse) o[axis] = 1 - o[axis]

    return xyz(o)
}

/**
 * GLSL coordinate expression used by gated propagation. The gate tensor is the
 * full volume, while the shader updates one slice, so the sweep-axis coordinate
 * is substituted with the slice uniform.
 */
function sliceCoord(
    x: string,
    y: string,
    z: string,
    permute: Permute
): string
{
    const coord: [string, string, string] = [z, y, x]
    coord[permute[0]] = 'slice'

    return xyz(coord)
}

/**
 * Initial margin pass: local paper-style difference construction.
 *
 * Computes local previous-z-face-minus-c111 margins before any long-range
 * propagation. The packed vec4 lanes are [c000, c010, c100, c110], matching
 * the four corners on the previous z face of the canonical trilinear cell.
 *
 * This corresponds to Delta_r(i) = V(i-r) - V(i) for the four incoming-face
 * offsets r. The original paper frames the skip test in terms of cell values
 * and ray order; this file keeps the same conservative meaning but represents
 * it as signed differences.
 */
class InitializeVertexMarginsProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        shape: Shape3,
        permute: Permute = [0, 1, 2],
        reverse: Reverse = []
    ) {
        const [depth, height, width] = shape

        this.outputShape = [depth, height, width, 2, 2]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool insideVolume(ivec3 p)
        {
            return all(greaterThanEqual(p, minCoords)) && all(lessThanEqual(p, maxCoords));
        }

        ivec3 outCoords()
        {
            ivec5 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        float volumeAt(ivec3 p)
        {
            return insideVolume(p) ? getA(p.z, p.y, p.x) : 0.0;
        }

        void main()
        {
            ivec3 p = outCoords();

            float v111 = volumeAt(p + ivec3(${voxelOffset( 0,  0,  0, permute, reverse)}));
            float v000 = volumeAt(p + ivec3(${voxelOffset(-1, -1, -1, permute, reverse)}));
            float v010 = volumeAt(p + ivec3(${voxelOffset(-1,  0, -1, permute, reverse)}));
            float v100 = volumeAt(p + ivec3(${voxelOffset( 0, -1, -1, permute, reverse)}));
            float v110 = volumeAt(p + ivec3(${voxelOffset( 0,  0, -1, permute, reverse)}));

            setOutput(vec4(v000, v010, v100, v110) - v111);
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
class PropagateVertexMarginsProgram implements GPGPUProgram 
{
    variableNames: string[]
    outputShape: number[]
    userCode: string

    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'slice', type: 'int' as const }]

    constructor(
        shape: PackedShape3,
        permute: Permute = [0, 1, 2],
        reverse: Reverse = [],
        gated = false
    ) {
        const [depth, height, width] = shape.slice(0, 3)

        this.outputShape = shape
        this.variableNames = gated ? ['A', 'B', 'C'] : ['A', 'B']

        const gateCode = gated ?
         `
        bvec4 gateAt(ivec3 p)
        {
            // The gate is indexed in the full 3D tensor. Replace the current
            // sweep-axis coordinate with the explicit slice being processed.
            p = ivec3(${sliceCoord('p.x', 'p.y', 'p.z', permute)});
            vec4 c = getC(p.z, p.y, p.x, 0, 0);
            return greaterThan(c, vec4(0.5));
        }
        `
        : 
        ``

        const applyGates = gated ? 
        `
        // open lane: clamp to positive, same as the forward pass.
        // closed lane: keep signed margins to prevent false reverse rejections.
        bvec4 open = gateAt(p);

        m111.x += open.x ? max(edgeBottleneck.x, 0.0) : edgeBottleneck.x;
        m111.y += open.y ? max(edgeBottleneck.y, 0.0) : edgeBottleneck.y;
        m111.z += open.z ? max(edgeBottleneck.z, 0.0) : edgeBottleneck.z;
        m111.w += open.w ? max(edgeBottleneck.w, 0.0) : edgeBottleneck.w;
        `
        : 
        `
        m111 += max(edgeBottleneck, 0.0);
        `

        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        float min4(vec4 v)
        {
            return min(min(min(v.x, v.y), v.z), v.w);
        }

        bool insideVolume(ivec3 p)
        {
            return all(greaterThanEqual(p, minCoords)) && all(lessThanEqual(p, maxCoords));
        }

        ivec3 outCoords()
        {
            ivec5 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        vec4 currentSliceAt(ivec3 p)
        {
            return getA(p.z, p.y, p.x, 0, 0);
        }

        vec4 previousSliceAt(ivec3 p)
        {
            return insideVolume(p) ? getB(p.z, p.y, p.x, 0, 0) : vec4(0.0);
        }

        ${gateCode}

        void main()
        {
            ivec3 p = outCoords();

            vec4 m111 =  currentSliceAt(p + ivec3(${sliceOffset( 0,  0,  0, permute, reverse)}));
            vec4 m110 = previousSliceAt(p + ivec3(${sliceOffset( 0,  0, -1, permute, reverse)}));
            vec4 m100 = previousSliceAt(p + ivec3(${sliceOffset( 0, -1, -1, permute, reverse)}));
            vec4 m010 = previousSliceAt(p + ivec3(${sliceOffset(-1,  0, -1, permute, reverse)}));
            vec4 m000 = previousSliceAt(p + ivec3(${sliceOffset(-1, -1, -1, permute, reverse)}));

            vec4 edgeBottleneck = vec4(min4(m000), min4(m010), min4(m100), min4(m110));

            ${applyGates}

            setOutput(m111);
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
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = false
    customUniforms = [{ name: 'tolerance', type: 'float' as const }]

    constructor(
        shape: Shape3,
        permute: Permute = [0, 1, 2],
        reverse: Reverse = []
    ) {
        const [depth, height, width] = shape
        this.outputShape = shape.map((n) => n + 1)

        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        bool insideVolume(ivec3 p)
        {
            return all(greaterThanEqual(p, minCoords)) && all(lessThanEqual(p, maxCoords));
        }

        ivec3 outCoords()
        {
            ivec3 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        vec4 marginsAt(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.z, p.y, p.x, 0, 0);
        }

        bool vertexShadow(ivec3 vertexCoords)
        {   
            vec4 vertexMargins = marginsAt(vertexCoords);
            return all(greaterThan(vertexMargins, vec4(-tolerance)));
        }

        bool cellShadow(ivec3 cellCoords)
        {
            ivec3 base = cellCoords - 1;

            // z=1 face of the 2x2x2 cell in canonical sweep
            // space. These are the four propagated margin entries that can
            // shadow the current cell.

            return
               vertexShadow(base + ivec3(${cellOffset(1, 1, 1, permute, reverse)})) &&
               vertexShadow(base + ivec3(${cellOffset(1, 0, 1, permute, reverse)})) &&
               vertexShadow(base + ivec3(${cellOffset(0, 1, 1, permute, reverse)})) &&
               vertexShadow(base + ivec3(${cellOffset(0, 0, 1, permute, reverse)}));
        }

        void main()
        {
            setOutput(float(cellShadow(outCoords())));
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
class computeEdgeGatesProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        shape: Shape3,
        permute: Permute = [0, 1, 2],
        reverse: Reverse = []
    ) {
        const [depth, height, width] = shape
        this.outputShape = shape.map((n) => n - 1).concat([2, 2]) as PackedShape3

        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        ivec3 outCoords()
        {
            ivec5 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        float shadowAt(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.z, p.y, p.x);
        }

        void main()
        {
            ivec3 p = outCoords();

            // Sample the same four canonical lanes used by the margin tensor.
            float s000 = shadowAt(p + ivec3(${cellOffset(-1, -1, -1, permute, reverse)}));
            float s010 = shadowAt(p + ivec3(${cellOffset(-1,  0, -1, permute, reverse)}));
            float s100 = shadowAt(p + ivec3(${cellOffset( 0, -1, -1, permute, reverse)}));
            float s110 = shadowAt(p + ivec3(${cellOffset( 0,  0, -1, permute, reverse)}));

            // Cell-mask values are 0/1. Invert them to get open/closed gates.
            vec4 edgeGates = 1.0 - vec4(s000, s010, s100, s110);
            
            setOutput(edgeGates);
        }
        `
    }
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
export function computeVertexMargins(
    volume: tf.Tensor3D,
    permute: Permute,
    reverse: Reverse,
    verbose: boolean = false
): tf.Tensor5D
{
    const axis = permute[0]
    const backwards = reverse.includes(axis)

    const initialMarginsProgram = new InitializeVertexMarginsProgram(volume.shape as Shape3, permute, reverse)
    let margins = runWebGLProgram(initialMarginsProgram, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logMean('initialMargins', margins)

    const slices = unstack3dPacked(margins, axis)
    const shape = slices[0].shape as PackedShape3
    const propagate = new PropagateVertexMarginsProgram(shape, permute, reverse)
    tf.dispose(margins)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step)
    {
        const next = runWebGLProgram( propagate, [slices[i], slices[i-step]], 'float32', [[i]], true)
        tf.dispose(slices[i])
        slices[i] = next
    }

    margins = stack3dPacked(slices, axis) as tf.Tensor5D
    tf.dispose(slices)
    if (verbose) logMean('margins', margins)

    return margins
}

export function computeVertexMinmax(
    volume: tf.Tensor3D,
    permute: Permute,
    reverse: Reverse,
    verbose: boolean = false
): tf.Tensor3D
{
    const margins = computeVertexMargins(volume, permute, reverse)
    if (verbose) logMean('margins', margins)

    const minmax = tf.tidy(() => margins.min([3, 4]).maximum(0).add(volume))
    tf.dispose(margins)
    if (verbose) logMean('minmax', minmax)

    return minmax as tf.Tensor3D
}

/**
 * Reverse-sweep version of computeMargins. It uses the forward-pass gates to
 * decide whether a route should extend only positive rejection margins or pass
 * signed margins through unchanged.
 */
export function computeGatedVertexMargins(
    volume: tf.Tensor3D,
    gates: tf.Tensor5D,
    permute: Permute,
    reverse: Reverse,
    verbose: boolean = false
): tf.Tensor5D
{
    const axis = permute[0]
    const backwards = reverse.includes(axis)

    const initialMarginsProgram = new InitializeVertexMarginsProgram(volume.shape as Shape3, permute, reverse)
    let margins = runWebGLProgram(initialMarginsProgram, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logMean('initialMargins', margins)

    const slices = unstack3dPacked(margins, axis)
    const shape = slices[0].shape as PackedShape3
    const propagate = new PropagateVertexMarginsProgram(shape, permute, reverse, true)
    tf.dispose(margins)

    const start = backwards ? slices.length - 3 : 2
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step)
    {
        const next = runWebGLProgram( propagate, [slices[i], slices[i-step], gates], 'float32', [[i]], true)

        tf.dispose(slices[i])
        slices[i] = next
    }

    margins = stack3dPacked(slices, axis) as tf.Tensor5D
    tf.dispose(slices)
    if (verbose) logMean('gatedMargins', margins)

    return margins
}

/**
 * Converts propagated vertex margins into a binary 3D cell mask.
 */
export function computeCellShadows(
    margins: tf.Tensor5D,
    permute: Permute,
    reverse: Reverse,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = margins.shape.slice(0, 3) as Shape3
    const program = new ComputeCellShadowsProgram(shape, permute, reverse)

    const shadowsMask = runWebGLProgram(program, [margins], 'float32', [[tolerance]], true) as tf.Tensor3D
    if (verbose) logMean('shadowsMask', shadowsMask)

    return shadowsMask
}

/**
 * Creates the per-lane open/closed mask used by reverse propagation.
 */
export function computeEdgeGates(
    forwardMask: tf.Tensor3D,
    permute: Permute,
    reverse: Reverse,
    verbose: boolean = false
): tf.Tensor5D
{
    const program = new computeEdgeGatesProgram(forwardMask.shape as Shape3, permute, reverse)
    const gates = runWebGLProgram(program, [forwardMask], 'float32', [], true) as tf.Tensor5D

    if (verbose) logMean('gates', gates)

    return gates
}

/**
 * Computes one directed cell mask for one dominant axis and octant.
 *
 * This is the core public operation: initialize local margins, propagate them
 * along the selected direction, then classify rejected cells.
 */
export function computeUnidirectionalShadowMap(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const { permute, reverse } = dominantAxisOctantToPermuteReverse(dominantAxis, octant)

    const margins = computeVertexMargins(volume, permute, reverse, verbose)
    const shadowsMask = computeCellShadows(margins, permute, reverse, tolerance, verbose)
    tf.dispose(margins)

    return shadowsMask
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
    dominantAxis: Axis,
    octant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const { permute, reverse } = dominantAxisOctantToPermuteReverse(dominantAxis, octant)
    const backwardReverse = complementReverse(reverse)

    const forwardShadows = computeUnidirectionalShadowMap(volume, dominantAxis, octant, tolerance, verbose)
    const backwardGates = computeEdgeGates(forwardShadows, permute, backwardReverse, verbose)

    const backwardMargins = computeGatedVertexMargins(volume, backwardGates, permute, backwardReverse, verbose)
    tf.dispose(backwardGates)

    const backwardShadows = computeCellShadows(backwardMargins, permute, backwardReverse, tolerance, verbose)
    tf.dispose(backwardMargins)

    const bidirectionalShadows = tf.maximum(forwardShadows, backwardShadows) as tf.Tensor3D
    tf.dispose([forwardShadows, backwardShadows])
    if (verbose) logMean('bidirectionalShadowsMask', bidirectionalShadows)

    return bidirectionalShadows
}

/**
 * Computes a bidirectional cell mask and reduces it into blocks. Binary min
 * pooling marks a block rejected only if every covered cell-mask entry is
 * rejected, preserving the conservative guarantee of the section 2.2 culling
 * test at a coarser traversal level.
 */
export function computeBidirectionalBlockShadowMap(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shadows = computeBidirectionalShadowMap(volume, dominantAxis, octant, tolerance, verbose)
    if (blockSize === 1) return shadows

    const blockShadows = minPool3d(shadows, blockSize, blockSize, 'same') as tf.Tensor3D
    tf.dispose(shadows)
    if (verbose) logMean('blockShadowsMask', blockShadows)

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
