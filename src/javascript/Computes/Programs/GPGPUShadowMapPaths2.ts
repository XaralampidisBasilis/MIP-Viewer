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
    axisToIndex,
    reverseOctant,
    applyPermutation,
    dominantAxisOctantToPermuteReverse,
} from '../../Utils/ShadowMapUtils'
import { minPool3d } from './pool3d'

type Shape3 = [number, number, number]
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
    octant: Octant,
): string
{
    const { permute, reverse } = dominantAxisOctantToPermuteReverse(axis, octant)

    const zyx = applyPermutation([z, y, x], permute)

    for (const dim of reverse) zyx[dim] = -zyx[dim]

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
    octant: Octant,
): string
{
    const { permute, reverse } = dominantAxisOctantToPermuteReverse(axis, octant)

    const zyx = applyPermutation([z, y, x], permute)

    for (const dim of reverse) zyx[dim] = -zyx[dim]

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
    octant: Octant,
): string
{
    const { permute, reverse } = dominantAxisOctantToPermuteReverse(axis, octant)

    const zyx = applyPermutation([z, y, x], permute)

    for (const dim of reverse) zyx[dim] = 1 - zyx[dim]

    return xyz(zyx)
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
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false
    customUniforms = [{ name: 'slice', type: 'int' as const }]

    constructor(
        shape: Shape3,
        axis: Axis = 'z',
        octant: Octant = '+++',
    ) {
        const [depth, height, width] = shape

        this.outputShape = shape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});
       
        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        bool insideVolume(ivec3 p)
        {
            if (p.x < minCoords.x || p.x > maxCoords.x) return false;
            if (p.y < minCoords.y || p.y > maxCoords.y) return false;
            if (p.z < minCoords.z || p.z > maxCoords.z) return false;

            return true;
        }

        ivec3 outputCoords()
        {
            ivec3 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        float currentSliceAt(ivec3 p)
        {
            return getA(p.z, p.y, p.x);
        }

        float previousSliceAt(ivec3 p)
        {            
            return insideVolume(p) ? getB(p.z, p.y, p.x) : 0.0;
        }

        float vertexMinmax(ivec3 p)
        {
            float v111 =  currentSliceAt(p + ivec3(${sliceOffset( 0,  0,  0, axis, octant)}));
            float v110 = previousSliceAt(p + ivec3(${sliceOffset( 0,  0, -1, axis, octant)}));
            float v100 = previousSliceAt(p + ivec3(${sliceOffset( 0, -1, -1, axis, octant)}));
            float v010 = previousSliceAt(p + ivec3(${sliceOffset(-1,  0, -1, axis, octant)}));
            float v000 = previousSliceAt(p + ivec3(${sliceOffset(-1, -1, -1, axis, octant)}));

            float bottleneck = min4(v000, v010, v100, v110);

            return max(v111, bottleneck);
        }

        void main()
        {
            setOutput(vertexMinmax(outputCoords()));
        }
        `
    }
}

class ComputeVertexMarginsProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        shape: Shape3,
        axis: Axis = 'z',
        octant: Octant = '+++',
    ) {
        const [depth, height, width] = shape
    
        this.outputShape = [depth, height, width, 2, 2] 
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        bool insideVolume(ivec3 p)
        {
            if (p.x < minCoords.x || p.x > maxCoords.x) return false;
            if (p.y < minCoords.y || p.y > maxCoords.y) return false;
            if (p.z < minCoords.z || p.z > maxCoords.z) return false;

            return true;
        }

        ivec3 outputCoords()
        {
            ivec5 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        float volumeAt(ivec3 p)
        {
            return getA(p.z, p.y, p.x);
        }

        float minmaxAt(ivec3 p)
        {
            return insideVolume(p) ? getB(p.z, p.y, p.x) : 0.0;
        }

        vec4 vertexMargins(ivec3 p)
        {
            float v111 = volumeAt(p + ivec3(${voxelOffset( 0,  0,  0, axis, octant)}));
            float v110 = minmaxAt(p + ivec3(${voxelOffset( 0,  0, -1, axis, octant)}));
            float v100 = minmaxAt(p + ivec3(${voxelOffset( 0, -1, -1, axis, octant)}));
            float v010 = minmaxAt(p + ivec3(${voxelOffset(-1,  0, -1, axis, octant)}));
            float v000 = minmaxAt(p + ivec3(${voxelOffset(-1, -1, -1, axis, octant)}));

            return (vec4(v000, v010, v100, v110) - v111);
        }

        void main()
        {
            setOutput(vertexMargins(outputCoords()));
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
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false
    customUniforms = [{ name: 'tolerance', type: 'float' as const }]

    constructor(
        shape: Shape3,
        axis: Axis = 'z',
        octant: Octant = '+++'
    ) {
        const [depth, height, width] = shape
    
        this.outputShape = shape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        bool insideVolume(ivec3 p)
        {
            if (p.x < minCoords.x || p.x > maxCoords.x) return false;
            if (p.y < minCoords.y || p.y > maxCoords.y) return false;
            if (p.z < minCoords.z || p.z > maxCoords.z) return false;

            return true;
        }

        ivec3 outputCoords()
        {
            ivec3 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        float volumeAt(ivec3 p)
        {
            return getA(p.z, p.y, p.x);
        }

        float minmaxAt(ivec3 p)
        {
            return insideVolume(p) ? getB(p.z, p.y, p.x) : 0.0;
        }

        bool vertexShadow(ivec3 p)
        {
            float v111 = volumeAt(p + ivec3(${voxelOffset( 0,  0,  0, axis, octant)}));
            float v110 = minmaxAt(p + ivec3(${voxelOffset( 0,  0, -1, axis, octant)}));
            float v100 = minmaxAt(p + ivec3(${voxelOffset( 0, -1, -1, axis, octant)}));
            float v010 = minmaxAt(p + ivec3(${voxelOffset(-1,  0, -1, axis, octant)}));
            float v000 = minmaxAt(p + ivec3(${voxelOffset(-1, -1, -1, axis, octant)}));

            float bottleneck = min4(v000, v010, v100, v110);
            float margin = v111 - bottleneck;

            return (margin < tolerance);
        }

        void main()
        {
            setOutput(float(vertexShadow(outputCoords())));
        }
        `
    }
}

class ComputeVertexHolesProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        shape: Shape3,
        axis: Axis = 'z',
        octant: Octant = '+++',
    ) {
        const [depth, height, width] = shape

        this.outputShape = shape.map((n) => n-1)
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        ivec3 outputCoords()
        {
            ivec3 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        bool cellShadow(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.z, p.y, p.x) > 0.5;
        }

        bool vertexHole(ivec3 coords)
        {
            return cellShadow(coords + ivec3(${cellOffset(1, 1, 1, axis, octant)}));
   
        }

        void main()
        {            
            setOutput(float(vertexHole(outputCoords())));
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
    packedInputs = false
    packedOutput = false

    constructor(
        shape: Shape3,
        axis: Axis = 'z',
        octant: Octant = '+++',
    ) {
        const [depth, height, width] = shape

        this.outputShape = shape.map((n) => n + 1)
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});
            
        bool insideVolume(ivec3 p)
        {
            if (p.x < minCoords.x || p.x > maxCoords.x) return false;
            if (p.y < minCoords.y || p.y > maxCoords.y) return false;
            if (p.z < minCoords.z || p.z > maxCoords.z) return false;

            return true;
        }

        ivec3 outputCoords()
        {
            ivec3 p = getOutputCoords();
            return ivec3(p.z, p.y, p.x);
        }

        bool vertexShadow(ivec3 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.z, p.y, p.x) > 0.5;
        }

        bool cellShadow(ivec3 coords)
        {
            ivec3 base = coords - 1;

            return (
                vertexShadow(base + ivec3(${cellOffset(1, 1, 1, axis, octant)})) &&
                vertexShadow(base + ivec3(${cellOffset(1, 0, 1, axis, octant)})) &&
                vertexShadow(base + ivec3(${cellOffset(0, 1, 1, axis, octant)})) &&
                vertexShadow(base + ivec3(${cellOffset(0, 0, 1, axis, octant)})) &&
                vertexShadow(base + ivec3(${cellOffset(1, 1, 0, axis, octant)})) &&
                vertexShadow(base + ivec3(${cellOffset(1, 0, 0, axis, octant)})) &&
                vertexShadow(base + ivec3(${cellOffset(0, 1, 0, axis, octant)})) 
            );
        }

        void main()
        {
            setOutput(float(cellShadow(outputCoords())));
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
export function computeVertexMinmax(
    volume: tf.Tensor3D,
    axis: Axis,
    octant: Octant,
    verbose: boolean = false
): tf.Tensor3D
{
    const { permute, reverse } = dominantAxisOctantToPermuteReverse(axis, octant)
    const dimension = permute[0]
    const backwards = reverse.includes(dimension)
    
    const slices = unstack3d(volume, dimension)
    const shape = slices[0].shape as Shape3
    const propagate = new PropagateVertexMinmaxProgram(shape, axis, octant)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? 0 : slices.length - 1
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step)
    {
        const next = runWebGLProgram(propagate, [slices[i], slices[i-step]], 'float32', [[i]], true)
        tf.dispose(slices[i])
        slices[i] = next
    }

    const vertexMinmax = stack3d(slices, dimension) 
    tf.dispose(slices)
    if (verbose) logMean('vertexMinmax', vertexMinmax)

    return vertexMinmax as tf.Tensor3D
}

/**
 * Converts propagated vertex minmax values into margins.
 */
export function computeVertexMargins(
    volume: tf.Tensor3D,
    vertexMinmax: tf.Tensor3D,
    axis: Axis,
    octant: Octant,
    verbose: boolean = false
): tf.Tensor5D
{
    const program = new ComputeVertexMarginsProgram(volume.shape, axis, octant)
    const vertexMargins = runWebGLProgram(program, [volume, vertexMinmax], 'float32', [], true) 
    if (verbose) logMean('vertexMargins', vertexMargins)

    return vertexMargins as tf.Tensor5D
}

/**
 * Converts propagated vertex margins into a binary 3D cell mask.
 */
export function computeVertexShadows(
    volume: tf.Tensor3D,
    vertexMinmax: tf.Tensor3D,
    axis: Axis,
    octant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const program = new ComputeVertexShadowsProgram(volume.shape, axis, octant)
    const vertexShadows = runWebGLProgram(program, [volume, vertexMinmax], 'bool', [[tolerance]], true) 
    if (verbose) logMean('vertexShadows', vertexShadows)

    return vertexShadows as tf.Tensor3D
}

/**
 * Converts propagated vertex margins into a binary 3D cell mask.
 */
export function computeCellShadows(
    vertexShadows: tf.Tensor3D,
    axis: Axis,
    octant: Octant,
    verbose: boolean = false
): tf.Tensor3D
{
    const program = new ComputeCellShadowsProgram(vertexShadows.shape, axis, octant)
    const cellShadows = runWebGLProgram(program, [vertexShadows], 'bool', [], true) 
    if (verbose) logMean('cellShadows', cellShadows)

    return cellShadows as tf.Tensor3D
}

export function computeVertexHoles(
    cellShadows: tf.Tensor3D,
    axis: Axis,
    octant: Octant,
    verbose: boolean = false
): tf.Tensor3D
{
    const program = new ComputeVertexHolesProgram(cellShadows.shape, axis, octant)
    const vertexHoles = runWebGLProgram(program, [cellShadows], 'bool', [], true) 
    if (verbose) logMean('vertexHoles', vertexHoles)

    return vertexHoles as tf.Tensor3D
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
    octant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const vertexMinmax = computeVertexMinmax(volume, axis, octant, verbose)
    const vertexShadows = computeVertexShadows(volume, vertexMinmax, axis, octant, tolerance, verbose)
    tf.dispose(vertexMinmax)

    const cellShadows = computeCellShadows(vertexShadows, axis, octant, verbose)
    tf.dispose(vertexShadows)

    return cellShadows
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
    octant: Octant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const backwardOctant = reverseOctant(octant)
    const forwardShadows = computeUnidirectionalShadowMap(volume, axis, octant, tolerance, verbose)

    const vertexHoles = computeVertexHoles(forwardShadows, axis, backwardOctant, verbose)
    const hollowVolume = tf.where(vertexHoles, 0, volume) 
    tf.dispose(vertexHoles)

    const backwardShadows = computeUnidirectionalShadowMap(hollowVolume, axis, backwardOctant, tolerance, verbose)
    tf.dispose(hollowVolume)

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
    octant: Octant,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor3D
{
    const shadowMap = computeBidirectionalShadowMap(volume, axis, octant, tolerance, verbose)
    if (blockSize === 1) return shadowMap

    const blockShadowMap = minPool3d(shadowMap, blockSize, blockSize, 'same') as tf.Tensor3D
    tf.dispose(shadowMap)
    if (verbose) logMean('blockShadowMap', blockShadowMap)

    return blockShadowMap
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
