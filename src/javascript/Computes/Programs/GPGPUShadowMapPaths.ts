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
import { stack3d } from './stack_keepDims_webgl'
import { unstack3d } from './unstack_keepDims_webgl'
import {
    type Axis,
    type Octant,
    type Permute,
    type Reverse,
    applyPermutation,
    complementReverse,
    inversePermutation,
    permuteReverseFromDominantAxisOctant,
    reverseOctant,
} from '../../Utils/ShadowMapUtils'
import { minPool3d } from './pool3d'

type Shape3 = [number, number, number]
type PackedShape3 = [number, number, number, 2, 2]
type Triple<T> = [T, T, T]
type Quad<T> = [T, T, T, T]
type CoordExpr = number | string

// In the comments below, a "ray class" means a dominant-axis/octant group.
// The renderer can select a class from the view ray without changing the
// precomputed texture layout.
const AXES: Triple<Axis> = ['x', 'y', 'z']

/**
 * A bidirectional sweep covers both signs along the dominant axis. That leaves
 * four representative octants per dominant axis, which together form the 12
 * cell masks packed by computeExtendedAnisotropicBidirectionalShadowMap.
 */
const OCTANTS: Record<Axis, Quad<Octant>> =
{
    x: ['+++', '+-+', '++-', '+--'],
    y: ['+++', '-++', '++-', '-+-'],
    z: ['+++', '+-+', '-++', '--+'],
}

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
class PropagateMinmaxProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'slice', type: 'int' as const }]

    constructor(
        shape: Shape3,
        permute: Permute = [0, 1, 2],
        reverse: Reverse = []
    ) {
        const [depth, height, width] = shape.slice(0, 3)

        this.outputShape = shape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});
       
        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        ivec3 outCoords()
        {
            ivec5 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        float currentSliceAt(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.z, p.y, p.x);
        }

        float previousSliceAt(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getB(p.z, p.y, p.x);
        }

        void main()
        {
            ivec3 p = outCoords();

            float v111 =  currentSliceAt(p + ivec3(${sliceOffset( 0,  0,  0, permute, reverse)}));
            float v110 = previousSliceAt(p + ivec3(${sliceOffset( 0,  0, -1, permute, reverse)}));
            float v100 = previousSliceAt(p + ivec3(${sliceOffset( 0, -1, -1, permute, reverse)}));
            float v010 = previousSliceAt(p + ivec3(${sliceOffset(-1,  0, -1, permute, reverse)}));
            float v000 = previousSliceAt(p + ivec3(${sliceOffset(-1, -1, -1, permute, reverse)}));

            float bottleneck = min4(v000, v010, v100, v110);
            float minmax = max(v111, bottleneck);

            setOutput(minmax);
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
class ClassifyShadowsProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = false
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

        ivec3 outCoords()
        {
            ivec3 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        float volumeAt(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.z, p.y, p.x);
        }

        float minmaxAt(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getB(p.z, p.y, p.x);
        }

        bool shadowAt(ivec3 p)
        {
            return (minmaxAt(p) - volumeAt(p) > -tolerance);
        }

        bool shadowed(ivec3 coords)
        {
            ivec3 base = coords - 1;

            return
                shadowAt(base + ivec3(${cellOffset(1, 1, 1, permute, reverse)})) &&
                shadowAt(base + ivec3(${cellOffset(1, 0, 1, permute, reverse)})) &&
                shadowAt(base + ivec3(${cellOffset(0, 1, 1, permute, reverse)})) &&
                shadowAt(base + ivec3(${cellOffset(0, 0, 1, permute, reverse)}));
        }

        void main()
        {
            setOutput(float(shadowed(outCoords())));
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
class BackwardGateProgram implements GPGPUProgram
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
            setOutput(1.0 - vec4(s000, s010, s100, s110));
        }
        `
    }
}

/**
 * Gated propagation pass for the reverse sweep.
 *
 * During bidirectional construction, the forward cell mask becomes the gate for
 * the reverse pass. Open gates behave like normal propagation. Closed gates
 * let the raw limiting margin through so the reverse pass does not double-count
 * blockers through a region already rejected by the forward pass.
 *
 * This implements the paper's bidirectional idea in difference space: compute
 * one directional cell mask, then evaluate the complementary direction while
 * preventing already-discarded cells from becoming artificial occluders.
 */
class PropagateGatedMarginsProgram implements GPGPUProgram
{
    variableNames = ['A', 'B', 'C']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'slice', type: 'int' as const }]

    constructor(
        shape: PackedShape3,
        permute: Permute = [0, 1, 2],
        reverse: Reverse = []
    ) {
        const [depth, height, width] = shape.slice(0, 3)

        this.outputShape = shape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        float min4(vec4 v)
        {
            return min(min(min(v.x, v.y), v.z), v.w);
        }

        ivec3 outCoords()
        {
            ivec5 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        vec4 currSliceAt(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.z, p.y, p.x, 0, 0);
        }

        vec4 prevSliceAt(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getB(p.z, p.y, p.x, 0, 0);
        }

        bvec4 gateAt(ivec3 p)
        {
            // The gate is indexed in the full 3D tensor. Replace the current
            // sweep-axis coordinate with the explicit slice being processed.
            p = ivec3(${sliceCoord('p.x', 'p.y', 'p.z', permute)});
            vec4 c = getC(p.z, p.y, p.x, 0, 0);
            return greaterThan(c, vec4(0.5));
        }

        void main()
        {
            ivec3 p = outCoords();
            bvec4 open = gateAt(p);

            vec4 m111 = currSliceAt(p + ivec3(${sliceOffset( 0,  0,  0, permute, reverse)}));
            vec4 m110 = prevSliceAt(p + ivec3(${sliceOffset( 0,  0, -1, permute, reverse)}));
            vec4 m100 = prevSliceAt(p + ivec3(${sliceOffset( 0, -1, -1, permute, reverse)}));
            vec4 m010 = prevSliceAt(p + ivec3(${sliceOffset(-1,  0, -1, permute, reverse)}));
            vec4 m000 = prevSliceAt(p + ivec3(${sliceOffset(-1, -1, -1, permute, reverse)}));

            vec4 n111 = vec4(
                min4(m000), 
                min4(m010), 
                min4(m100), 
                min4(m110)
            );

            // open lane: clamp to positive, same as the forward pass.
            // closed lane: keep signed margins to prevent false reverse rejections.

            m111.x += open.x ? max(n111.x, 0.0) : n111.x;
            m111.y += open.y ? max(n111.y, 0.0) : n111.y;
            m111.z += open.z ? max(n111.z, 0.0) : n111.z;
            m111.w += open.w ? max(n111.w, 0.0) : n111.w;

            setOutput(m111);
        }
        `
    }
}

/**
 * Binary OR of two 0/1 cell masks. Used to merge forward and reverse passes
 * into one bidirectional mask.
 */
class OrShadowsProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(shape: Shape3)
    {
        this.outputShape = shape
        this.userCode = `
        uvec4 bits(vec4 v)
        {
            return uvec4(round(v)) & 1u;
        }

        ivec3 outCoords()
        {
            ivec3 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        void main()
        {
            ivec3 p = outCoords();
            setOutput(vec4(bits(getA(p.z, p.y, p.x)) | bits(getB(p.z, p.y, p.x))));
        }
        `
    }
}

/**
 * Packs four ray classes for a single dominant axis into a 4-bit value per
 * packed lane. The output still has a 3D shape; only the value encoding
 * changes.
 */
class PackAxisProgram implements GPGPUProgram
{
    variableNames = ['A', 'B', 'C', 'D']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(shape: Shape3)
    {
        this.outputShape = shape
        this.userCode = `
        uvec4 bit(vec4 v)
        {
            return uvec4(round(v)) & 1u;
        }

        ivec3 outCoords()
        {
            ivec3 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        uvec4 pack4(uvec4 a, uvec4 b, uvec4 c, uvec4 d)
        {
            // Four representative octants for one dominant axis.
            return (a << 0u) | (b << 1u) | (c << 2u) | (d << 3u);
        }

        void main()
        {
            ivec3 p = outCoords();

            uvec4 a = bit(getA(p.z, p.y, p.x));
            uvec4 b = bit(getB(p.z, p.y, p.x));
            uvec4 c = bit(getC(p.z, p.y, p.x));
            uvec4 d = bit(getD(p.z, p.y, p.x));

            setOutput(vec4(pack4(a, b, c, d)));
        }
        `
    }
}

/**
 * Packs three dominant-axis nibbles into one 12-bit value per packed lane.
 * This is the texture-resident form of the anisotropic directional fields: one
 * lookup can recover the precomputed skip/reject information for the relevant
 * ray class.
 *
 * Half-float textures exactly cover integers in [-2048, 2048], so the unsigned
 * 0..4095 packed value is shifted down before storage.
 */
class PackExtendedProgram implements GPGPUProgram
{
    variableNames = ['A', 'B', 'C']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(shape: Shape3)
    {
        this.outputShape = shape
        this.userCode = `
        uvec4 nibble(vec4 v)
        {
            return uvec4(round(v)) & 15u;
        }

        ivec3 outCoords()
        {
            ivec3 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        ivec4 pack12(uvec4 x, uvec4 y, uvec4 z)
        {
            uvec4 p = (x << 0u) | (y << 4u) | (z << 8u);
            return ivec4(p) - ivec4(2048);
        }

        void main()
        {
            ivec3 p = outCoords();

            uvec4 x = nibble(getA(p.z, p.y, p.x));
            uvec4 y = nibble(getB(p.z, p.y, p.x));
            uvec4 z = nibble(getC(p.z, p.y, p.x));

            setOutput(vec4(pack12(x, y, z)));
        }
        `
    }
}

/**
 * Debug helper program: extracts one bit from a packed single-axis cell mask.
 */
class UnpackAxisProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'maskBit', type: 'int' as const }]

    constructor(shape: Shape3)
    {
        this.outputShape = shape
        this.userCode = `
        uvec4 nibble(vec4 v)
        {
            return uvec4(round(v)) & 15u;
        }

        ivec3 outCoords()
        {
            ivec3 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        void main()
        {
            ivec3 p = outCoords();
            uvec4 packed = nibble(getA(p.z, p.y, p.x));
            setOutput(vec4((packed >> maskBit) & 1u));
        }
        `
    }
}

/**
 * Debug helper program: extracts one bit from the final 12-bit packed mask.
 */
class UnpackExtendedProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'maskBit', type: 'int' as const }]

    constructor(shape: Shape3)
    {
        this.outputShape = shape
        this.userCode = `
        ivec4 stored(vec4 v)
        {
            return clamp(ivec4(round(v)), -2048, 2047);
        }

        ivec3 outCoords()
        {
            ivec3 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        void main()
        {
            ivec3 p = outCoords();
            uvec4 packed = uvec4(stored(getA(p.z, p.y, p.x)) + ivec4(2048));
            setOutput(vec4((packed >> maskBit) & 1u));
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
function propagateMinmax(
    volume: tf.Tensor3D,
    permute: Permute,
    reverse: Reverse,
    verbose: boolean = false
): tf.Tensor5D
{
    const axis = permute[0]
    const backwards = reverse.includes(axis)

    const slices = unstack3d(volume, axis)

    const shape = slices[0].shape as Shape3
    const propagate = new PropagateMinmaxProgram(shape, permute, reverse)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    // March in the direction implied by reverse. The first slice already has
    // its local vertex minmax; every following slice consumes the prior
    // propagated slice.
    for (let i = start; i !== end; i += step)
    {
        const next = runWebGLProgram( propagate, [slices[i], slices[i - step]], 'float32', [[i]], true)

        tf.dispose(slices[i])
        slices[i] = next
    }

    const minmax = stack3d(slices, axis) as tf.Tensor5D
    tf.dispose(slices)
    if (verbose) logMean('minmax', minmax)

    return minmax
}

/**
 * Reverse-sweep version of propagateMinmax. It uses the forward-pass gates to
 * decide whether a route should extend only positive rejection margins or pass
 * signed margins through unchanged.
 */
function computeGatedMargins(
    volume: tf.Tensor3D,
    gates: tf.Tensor5D,
    permute: Permute,
    reverse: Reverse,
    verbose: boolean = false
): tf.Tensor5D
{
    const axis = permute[0]
    const backwards = reverse.includes(axis)

    const initialMarginsProgram = new InitialMarginsProgram(volume.shape as Shape3, permute, reverse)
    let margins = runWebGLProgram(initialMarginsProgram, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logMean('initialMargins', margins)

    const slices = unstackPacked(margins, axis)
    tf.dispose(margins)

    const shape = slices[0].shape as PackedShape3
    const propagate = new PropagateGatedMarginsProgram(shape, permute, reverse)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    // Same dependency pattern as computeMargins, with the gate tensor supplied
    // to every slice update.
    for (let i = start; i !== end; i += step)
    {
        const next = runWebGLProgram( propagate, [slices[i], slices[i - step], gates], 'float32', [[i]], true)

        tf.dispose(slices[i])
        slices[i] = next
    }

    margins = stackPacked(slices, axis) as tf.Tensor5D
    tf.dispose(slices)
    if (verbose) logMean('gatedMargins', margins)

    return margins
}

/**
 * Converts propagated vertex margins into a binary 3D cell mask.
 */
function classifyShadowsMask(
    margins: tf.Tensor5D,
    permute: Permute,
    reverse: Reverse,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = margins.shape.slice(0, 3) as Shape3
    const program = new ClassifyShadowsProgram(shape, permute, reverse)

    const shadowsMask = runWebGLProgram(program, [margins], 'float32', [[tolerance]], true) as tf.Tensor3D
    if (verbose) logMean('shadowsMask', shadowsMask)

    return shadowsMask
}

/**
 * Creates the per-lane open/closed mask used by reverse propagation.
 */
function computeBackwardGates(
    forwardMask: tf.Tensor3D,
    permute: Permute,
    reverse: Reverse,
    verbose: boolean = false
): tf.Tensor5D
{
    const program = new BackwardGateProgram(forwardMask.shape as Shape3, permute, reverse)
    const gates = runWebGLProgram(program, [forwardMask], 'float32', [], true) as tf.Tensor5D

    if (verbose) logMean('gates', gates)

    return gates
}

/**
 * Merges forward and backward cell masks using a lane-wise binary OR.
 */
function orShadowsMask(
    a: tf.Tensor3D,
    b: tf.Tensor3D,
    verbose: boolean = false
): tf.Tensor3D
{
    const program = new OrShadowsProgram(a.shape as Shape3)
    const shadowsMask = runWebGLProgram(program, [a, b], 'float32', [], true) as tf.Tensor3D

    if (verbose) logMean('bidirectionalShadowsMask', shadowsMask)

    return shadowsMask
}

/**
 * Packs four directional cell masks belonging to one dominant axis.
 */
function packAxisMasks(
    masks: Quad<tf.Tensor3D>,
    verbose: boolean = false
): tf.Tensor3D
{
    const program = new PackAxisProgram(masks[0].shape as Shape3)
    const packed = runWebGLProgram(program, masks, 'float32', [], true) as tf.Tensor3D

    if (verbose) logMean('axisPack', packed)

    return packed
}

/**
 * Packs the x/y/z dominant-axis mask groups into the final anisotropic texture.
 */
function packExtendedMasks(
    masks: Triple<tf.Tensor3D>,
    verbose: boolean = false
): tf.Tensor3D
{
    const program = new PackExtendedProgram(masks[0].shape as Shape3)
    const packed = runWebGLProgram(program, masks, 'float32', [], true) as tf.Tensor3D

    if (verbose) logMean('extendedPack', packed)

    return packed
}

/**
 * Computes the four representative octants for one dominant axis and packs
 * them into a single axis mask group.
 */
function computeAxisPack(
    axis: Axis,
    computeMask: (octant: Octant) => tf.Tensor3D
): tf.Tensor3D
{
    const masks: tf.Tensor3D[] = []

    try
    {
        for (const octant of OCTANTS[axis])
        {
            masks.push(computeMask(octant))
        }

        return packAxisMasks(masks as Quad<tf.Tensor3D>)
    }
    finally
    {
        tf.dispose(masks)
    }
}

/**
 * Computes and packs the three dominant-axis groups.
 */
function computeExtendedPack(
    computeMask: (axis: Axis) => tf.Tensor3D,
    verbose: boolean
): tf.Tensor3D
{
    const masks: tf.Tensor3D[] = []

    try
    {
        for (const axis of AXES)
        {
            masks.push(computeMask(axis))
        }

        const packed = packExtendedMasks(masks as Triple<tf.Tensor3D>)
        if (verbose) logExtendedMaps(packed)

        return packed
    }
    finally
    {
        tf.dispose(masks)
    }
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
    const { permute, reverse } = permuteReverseFromDominantAxisOctant(dominantAxis, octant)

    const margins = computeMargins(volume, permute, reverse, verbose)
    const shadowsMask = classifyShadowsMask(margins, permute, reverse, tolerance, verbose)
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
    const { permute, reverse } = permuteReverseFromDominantAxisOctant(dominantAxis, octant)

    const forwardMask = computeUnidirectionalShadowMap(volume, dominantAxis, octant, tolerance, verbose)
    const backwardReverse = complementReverse(reverse)
    const gates = computeBackwardGates(forwardMask, permute, backwardReverse, verbose)
    const backwardMargins = computeGatedMargins(volume, gates, permute, backwardReverse, verbose)
    const backwardMask = classifyShadowsMask(backwardMargins, permute, backwardReverse, tolerance, verbose)
    const shadowsMask = orShadowsMask(forwardMask, backwardMask, verbose)

    tf.dispose([gates, backwardMargins, forwardMask, backwardMask])

    return shadowsMask
}

/**
 * Convenience wrapper for querying the opposite octant.
 */
export function computeBidirectionalShadowMapReverse(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    return computeBidirectionalShadowMap(volume, dominantAxis, reverseOctant(octant), tolerance, verbose)
}

/**
 * Computes the 12-direction extended texture using forward-only cell masks.
 */
export function computeExtendedAnisotropicUnidirectionalShadowMap(
    volume: tf.Tensor3D,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    return computeExtendedPack(
        (axis) => computeAxisPack(
            axis,
            (octant) => computeUnidirectionalShadowMap(volume, axis, octant, tolerance)
        ),
        verbose
    )
}

/**
 * Computes the full 12-direction extended anisotropic bidirectional texture.
 *
 * For each dominant axis, four representative octants are computed and packed.
 * Because each representative mask is bidirectional, the opposing octants are
 * included implicitly.
 */
export function computeExtendedAnisotropicBidirectionalShadowMap(
    volume: tf.Tensor3D,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    return computeExtendedPack(
        (axis) => computeAxisPack(
            axis,
            (octant) => computeBidirectionalShadowMap(volume, axis, octant, tolerance)
        ),
        verbose
    )
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
    const shadowsMask = computeBidirectionalShadowMap(volume, dominantAxis, octant, tolerance, verbose)

    if (blockSize === 1) return shadowsMask

    const blocks = minPool3d(shadowsMask, blockSize, blockSize, 'same') as tf.Tensor3D
    if (verbose) logMean('blockShadowsMask', blocks)

    tf.dispose(shadowsMask)

    return blocks
}

/**
 * Computes the extended anisotropic bidirectional mask after block reduction of
 * each component mask.
 */
export function computeExtendedAnisotropicBidirectionalBlockShadowMap(
    volume: tf.Tensor3D,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    return computeExtendedPack(
        (axis) => computeAxisPack(
            axis,
            (octant) => computeBidirectionalBlockShadowMap(volume, axis, octant, tolerance, blockSize)
        ),
        verbose
    )
}

/**
 * Reference path for a unidirectional cell mask.
 *
 * This physically transforms the volume into canonical sweep orientation,
 * runs the same canonical implementation, then transforms the result back. It
 * is useful for checking the offset-injection path used by the fast version.
 */
export function computeUnidirectionalShadowMapReference(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const { permute, reverse } = permuteReverseFromDominantAxisOctant(dominantAxis, octant)

    const reversed = volume.reverse(reverse) as tf.Tensor3D
    const canonicalVolume = reversed.transpose(permute) as tf.Tensor3D
    tf.dispose(reversed)

    const margins = computeMargins(canonicalVolume, [0, 1, 2], [], verbose)
    const canonicalMask = classifyShadowsMask(margins, [0, 1, 2], [], tolerance, verbose)
    tf.dispose([margins, canonicalVolume])

    const unpermuted = canonicalMask.transpose(inversePermutation(permute)) as tf.Tensor3D
    tf.dispose(canonicalMask)

    const shadowsMask = unpermuted.reverse(reverse) as tf.Tensor3D
    tf.dispose(unpermuted)

    return shadowsMask
}

/**
 * Reference path for the forward-only extended anisotropic cell-mask texture.
 */
export function computeExtendedAnisotropicUnidirectionalShadowMapReference(
    volume: tf.Tensor3D,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    return computeExtendedPack(
        (axis) => computeAxisPack(
            axis,
            (octant) => computeUnidirectionalShadowMapReference(volume, axis, octant, tolerance)
        ),
        verbose
    )
}

/**
 * Debug view that keeps one real z/+++ mask and fills the rest with ones. This
 * is useful when validating final packing/unpacking or visualizing one channel
 * through the renderer.
 */
export function computeExtendedAnisotropicBidirectionalShadowMapSingular(
    volume: tf.Tensor3D,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const zMask = computeUnidirectionalShadowMap(volume, 'z', '+++', tolerance)
    const masks: tf.Tensor3D[] = []

    try
    {
        masks.push(packSyntheticAxis([tf.onesLike(zMask), tf.onesLike(zMask), tf.onesLike(zMask), tf.onesLike(zMask)]))
        masks.push(packSyntheticAxis([tf.onesLike(zMask), tf.onesLike(zMask), tf.onesLike(zMask), tf.onesLike(zMask)]))
        masks.push(packSyntheticAxis([tf.clone(zMask), tf.onesLike(zMask), tf.onesLike(zMask), tf.onesLike(zMask)]))

        const packed = packExtendedMasks(masks as Triple<tf.Tensor3D>)
        if (verbose) logExtendedMaps(packed)

        return packed
    }
    finally
    {
        tf.dispose(masks)
        tf.dispose(zMask)
    }
}

/**
 * Packs synthetic/debug component masks and disposes them after packing.
 */
function packSyntheticAxis(masks: Quad<tf.Tensor3D>): tf.Tensor3D
{
    try
    {
        return packAxisMasks(masks)
    }
    finally
    {
        tf.dispose(masks)
    }
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
 * Logs each unpacked bit from a single-axis packed mask.
 */
function logAxisMaps(mask: tf.Tensor3D): void
{
    const unpack = new UnpackAxisProgram(mask.shape as Shape3)

    for (let maskBit = 0; maskBit < 4; maskBit++)
    {
        console.log(
            `shadowsMask${maskBit}`,
            tf.tidy(() => runWebGLProgram(unpack, [mask], 'float32', [[maskBit]]).mean([0, 1, 2]).dataSync())
        )
    }
}

/**
 * Logs all 12 unpacked bits from the extended anisotropic mask.
 */
function logExtendedMaps(mask: tf.Tensor3D): void
{
    const unpack = new UnpackExtendedProgram(mask.shape as Shape3)
    const labels = [
        'shadowsMaskX0',
        'shadowsMaskX1',
        'shadowsMaskX2',
        'shadowsMaskX3',
        'shadowsMaskY0',
        'shadowsMaskY1',
        'shadowsMaskY2',
        'shadowsMaskY3',
        'shadowsMaskZ0',
        'shadowsMaskZ1',
        'shadowsMaskZ2',
        'shadowsMaskZ3',
    ]

    for (let maskBit = 0; maskBit < labels.length; maskBit++)
    {
        console.log(
            labels[maskBit],
            tf.tidy(() => runWebGLProgram(unpack, [mask], 'float32', [[maskBit]]).mean([0, 1, 2]).dataSync())
        )
    }
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
