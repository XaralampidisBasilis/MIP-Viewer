/**
 * 2D extended anisotropic shadow masks for maximum intensity projection.
 *
 * This is the 2D analogue of GPGPUShadowMapDifferencesDocumented.ts. The 3D
 * version works on trilinear volume cells. This file works on bilinear image
 * cells: each cell is a unit square with four corner samples c00, c10, c01,
 * and c11.
 *
 * The algorithm is written in one canonical orientation: rays propagate in
 * local +y, so y is the dominant sweep axis in the GLSL stencil. Any requested
 * dominant-axis/quadrant pair is converted into that canonical case by
 * permuting and reversing offsets before they are injected into the shaders.
 *
 * In the canonical initial-margin pass, p is the c11 corner. The two vertices
 * on the previous y edge, c00 and c10, are compared against c11. The packed
 * margin lanes store [c00 - c11, c10 - c11]. Positive means the incoming edge
 * can already dominate the target corner for a maximum-intensity projection.
 *
 * Internally, the final shadow-map values are described as cell masks: 1 means
 * the bilinear cell is conservatively rejected for a ray class, 0 means it can
 * still contribute to the MIP.
 */
import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

export type Axis2D = 'x' | 'y'
export type Sign2D = '+' | '-'
export type Dimension2D = 0 | 1
export type Quadrant = `${Sign2D}${Sign2D}`
export type Permute2D = [Dimension2D, Dimension2D]
export type Reverse2D = Dimension2D[]

type Shape2 = [number, number]
type PackedShape2 = [number, number, 2, 2]
type Pair<T> = [T, T]
type CoordExpr = number | string
type PadMode2D = 'valid' | 'same' | number

// TensorFlow stores 2D tensors as [y, x]. Quadrant strings are ordered as xy.
const AXES: Pair<Axis2D> = ['x', 'y']

/**
 * A bidirectional sweep covers both signs of the dominant axis. That leaves
 * two representative quadrants per dominant axis.
 */
const QUADRANTS: Record<Axis2D, Pair<Quadrant>> =
{
    x: ['++', '+-'],
    y: ['++', '-+'],
}

function xy(coordsYx: [CoordExpr, CoordExpr]): string
{
    return [coordsYx[1], coordsYx[0]].join(', ')
}

function reverseSign(sign: Sign2D): Sign2D
{
    return sign === '+' ? '-' : '+'
}

function reverseQuadrant(quadrant: Quadrant): Quadrant
{
    return `${reverseSign(quadrant[0] as Sign2D)}${reverseSign(quadrant[1] as Sign2D)}`
}

function complementReverse(reverse: Reverse2D): Reverse2D
{
    const set = new Set<Dimension2D>(reverse)
    const complement: Reverse2D = []

    for (const axis of [0, 1] as const)
    {
        if (!set.has(axis)) complement.push(axis)
    }

    return complement
}

function inversePermutation(permute: Permute2D): Permute2D
{
    const inv = new Array<number>(permute.length)

    for (let i = 0; i < permute.length; i++)
    {
        inv[permute[i]] = i
    }

    return inv as Permute2D
}

function applyPermutation(offsetYx: [number, number], permute: Permute2D): [number, number]
{
    const physical: [number, number] = [0, 0]

    physical[permute[0]] = offsetYx[0]
    physical[permute[1]] = offsetYx[1]

    return physical
}

function reverseFromQuadrant(quadrant: Quadrant): Reverse2D
{
    const reverse: Reverse2D = []

    if (quadrant[1] === '-') reverse.push(0) // y sign
    if (quadrant[0] === '-') reverse.push(1) // x sign

    return reverse
}

function permuteReverseFromDominantAxisQuadrant(
    dominantAxis: Axis2D,
    quadrant: Quadrant
): { permute: Permute2D, reverse: Reverse2D }
{
    const permute: Permute2D = dominantAxis === 'y' ? [0, 1] : [1, 0]
    const reverse = reverseFromQuadrant(quadrant)

    return { permute, reverse }
}

/**
 * Convert a canonical sweep-space offset into the physical tensor orientation.
 * The TypeScript side thinks in local x/y offsets. TensorFlow stores [y, x],
 * so applyPermutation works in [y, x] order and xy emits GLSL ivec2 arguments.
 */
function sampleOffset(
    x: number,
    y: number,
    permute: Permute2D,
    reverse: Reverse2D
): string
{
    const o = applyPermutation([y, x], permute)

    for (const axis of reverse) o[axis] = -o[axis]

    return xy(o)
}

/**
 * Offset used while propagating one 1D line. The neighbor from the previous
 * sweep line is supplied as a separate tensor, so movement along the sweep
 * axis is erased after orientation is applied.
 */
function lineOffset(
    x: number,
    y: number,
    permute: Permute2D,
    reverse: Reverse2D
): string
{
    const o = applyPermutation([y, x], permute)

    for (const axis of reverse) o[axis] = -o[axis]

    o[permute[0]] = 0

    return xy(o)
}

/**
 * Offset for cell-corner addressing. Reversal mirrors a unit cell corner, so
 * coordinate 0 becomes 1 and coordinate 1 becomes 0 instead of simply flipping
 * the sign.
 */
function cellOffset(
    x: number,
    y: number,
    permute: Permute2D,
    reverse: Reverse2D
): string
{
    const o = applyPermutation([y, x], permute)

    for (const axis of reverse) o[axis] = 1 - o[axis]

    return xy(o)
}

/**
 * GLSL coordinate expression used by gated propagation. The gate tensor is the
 * full image, while the shader updates one line, so the sweep-axis coordinate
 * is substituted with the slice uniform.
 */
function lineCoord(
    x: string,
    y: string,
    permute: Permute2D
): string
{
    const coord: [string, string] = [y, x]
    coord[permute[0]] = 'slice'

    return xy(coord)
}

/**
 * Initial margin pass for bilinear cells.
 *
 * The packed vec4 uses only x and y lanes:
 *
 *     x: c00 - c11
 *     y: c10 - c11
 *
 * z and w are intentionally kept at zero so the tensor can still use the same
 * convenient [2, 2] packed trailing dimensions as the 3D implementation.
 */
class InitialMarginsProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        shape: Shape2,
        permute: Permute2D = [0, 1],
        reverse: Reverse2D = []
    ) {
        const [height, width] = shape

        this.outputShape = [height, width, 2, 2]
        this.userCode = `
        const ivec2 minCoords = ivec2(0);
        const ivec2 maxCoords = ivec2(${width - 1}, ${height - 1});

        bool insideImage(ivec2 p)
        {
            return all(greaterThanEqual(p, minCoords)) && all(lessThanEqual(p, maxCoords));
        }

        ivec2 outCoords()
        {
            ivec4 p = getOutputCoords();
            return ivec2(p.y, p.x);
        }

        float imageAt(ivec2 p)
        {
            // Outside the image is treated as empty intensity, matching the
            // boundary convention used by the 3D program.
            return insideImage(p) ? getA(p.y, p.x) : 0.0;
        }

        void main()
        {
            ivec2 p = outCoords();

            // p is c11 in canonical cell space. c00 and c10 are the two
            // vertices on the previous y edge of the bilinear cell.
            float c11 = imageAt(p + ivec2(${sampleOffset( 0,  0, permute, reverse)}));
            float c00 = imageAt(p + ivec2(${sampleOffset(-1, -1, permute, reverse)}));
            float c10 = imageAt(p + ivec2(${sampleOffset( 0, -1, permute, reverse)}));

            setOutput(vec4(c00 - c11, c10 - c11, 0.0, 0.0));
        }
        `
    }
}

/**
 * Dynamic-programming propagation pass.
 *
 * A contains the current line's local margins. B contains the already
 * propagated previous line. For each incoming edge vertex r:
 *
 *     Delta_r(i) += max(min_s Delta_s(i-r), 0)
 *
 * In 2D the incoming edge has two vertices, so min_s is a min over the two
 * active lanes of the predecessor margin vector.
 */
class PropagateMarginsProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'slice', type: 'int' as const }]

    constructor(
        shape: PackedShape2,
        permute: Permute2D = [0, 1],
        reverse: Reverse2D = []
    ) {
        const [height, width] = shape.slice(0, 2)

        this.outputShape = shape
        this.userCode = `
        const ivec2 minCoords = ivec2(0);
        const ivec2 maxCoords = ivec2(${width - 1}, ${height - 1});

        float min2(vec4 v)
        {
            return min(v.x, v.y);
        }

        ivec2 outCoords()
        {
            ivec4 p = getOutputCoords();
            return ivec2(p.y, p.x);
        }

        vec4 localMargins(ivec2 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.y, p.x, 0, 0);
        }

        vec4 propagatedMargins(ivec2 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getB(p.y, p.x, 0, 0);
        }

        void main()
        {
            ivec2 p = outCoords();

            vec4 margins = localMargins(p + ivec2(${lineOffset( 0,  0, permute, reverse)}));
            vec4 pred00 = propagatedMargins(p + ivec2(${lineOffset(-1, -1, permute, reverse)}));
            vec4 pred10 = propagatedMargins(p + ivec2(${lineOffset( 0, -1, permute, reverse)}));

            vec2 carry = max(vec2(min2(pred00), min2(pred10)), vec2(0.0));
            setOutput(vec4(margins.xy + carry, 0.0, 0.0));
        }
        `
    }
}

/**
 * Builds the reverse-pass gate tensor from the forward cell mask.
 *
 * Each active lane is 1 when the corresponding forward edge vertex is not
 * rejected. The reverse propagation shader reads this as "the reverse
 * rejection is allowed to grow through this lane".
 */
class BackwardGateProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        shape: Shape2,
        permute: Permute2D = [0, 1],
        reverse: Reverse2D = []
    ) {
        const [height, width] = shape
        this.outputShape = shape.map((n) => n - 1).concat([2, 2]) as PackedShape2

        this.userCode = `
        const ivec2 minCoords = ivec2(0);
        const ivec2 maxCoords = ivec2(${width - 1}, ${height - 1});

        ivec2 outCoords()
        {
            ivec4 p = getOutputCoords();
            return ivec2(p.y, p.x);
        }

        float cellMask(ivec2 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.y, p.x);
        }

        void main()
        {
            ivec2 p = outCoords();

            float mask00 = cellMask(p + ivec2(${cellOffset(-1, -1, permute, reverse)}));
            float mask10 = cellMask(p + ivec2(${cellOffset( 0, -1, permute, reverse)}));

            setOutput(vec4(1.0 - mask00, 1.0 - mask10, 0.0, 0.0));
        }
        `
    }
}

/**
 * Gated propagation pass for the reverse sweep.
 *
 * Open gates behave like normal propagation. Closed gates let the signed
 * limiting margin through so the reverse pass does not double-count blockers
 * through a region already rejected by the forward pass.
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
        shape: PackedShape2,
        permute: Permute2D = [0, 1],
        reverse: Reverse2D = []
    ) {
        const [height, width] = shape.slice(0, 2)

        this.outputShape = shape
        this.userCode = `
        const ivec2 minCoords = ivec2(0);
        const ivec2 maxCoords = ivec2(${width - 1}, ${height - 1});

        float min2(vec4 v)
        {
            return min(v.x, v.y);
        }

        ivec2 outCoords()
        {
            ivec4 p = getOutputCoords();
            return ivec2(p.y, p.x);
        }

        vec4 localMargins(ivec2 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.y, p.x, 0, 0);
        }

        vec4 propagatedMargins(ivec2 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getB(p.y, p.x, 0, 0);
        }

        vec4 gateAt(ivec2 p)
        {
            // The gate is indexed in the full 2D tensor. Replace the current
            // sweep-axis coordinate with the explicit line being processed.
            p = ivec2(${lineCoord('p.x', 'p.y', permute)});
            return getC(p.y, p.x, 0, 0);
        }

        void main()
        {
            ivec2 p = outCoords();
            bvec2 open = greaterThan(gateAt(p).xy, vec2(0.5));

            vec4 margins = localMargins(p + ivec2(${lineOffset( 0,  0, permute, reverse)}));
            vec4 pred00 = propagatedMargins(p + ivec2(${lineOffset(-1, -1, permute, reverse)}));
            vec4 pred10 = propagatedMargins(p + ivec2(${lineOffset( 0, -1, permute, reverse)}));

            vec2 carry = vec2(min2(pred00), min2(pred10));
            vec2 propagated = margins.xy;

            propagated.x += open.x ? max(carry.x, 0.0) : carry.x;
            propagated.y += open.y ? max(carry.y, 0.0) : carry.y;

            setOutput(vec4(propagated, 0.0, 0.0));
        }
        `
    }
}

/**
 * Converts propagated vertex margins into a binary cell mask.
 *
 * The margin tensor stores vec4 values but only the first two lanes are active:
 * margins to c00 and c10 on the incoming edge. A cell is rejected only when the
 * two relevant target-edge margin entries both prove dominance.
 */
class ClassifyCellMaskProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = false
    customUniforms = [{ name: 'tolerance', type: 'float' as const }]

    constructor(
        shape: Shape2,
        permute: Permute2D = [0, 1],
        reverse: Reverse2D = []
    ) {
        const [height, width] = shape
        this.outputShape = shape.map((n) => n + 1)

        this.userCode = `
        const ivec2 minCoords = ivec2(0);
        const ivec2 maxCoords = ivec2(${width - 1}, ${height - 1});

        ivec2 outCoords()
        {
            ivec2 p = getOutputCoords();
            return ivec2(p.y, p.x);
        }

        vec4 vertexMargins(ivec2 p)
        {
            p = clamp(p, minCoords, maxCoords);
            return getA(p.y, p.x, 0, 0);
        }

        bool positive(vec4 m)
        {
            return all(greaterThan(m.xy + vec2(tolerance), vec2(0.0)));
        }

        bool cellRejected(ivec2 cellCoords)
        {
            ivec2 base = cellCoords - 1;

            // y=1 edge of the 2x2 cell neighborhood in canonical sweep space.
            return
                positive(vertexMargins(base + ivec2(${cellOffset(1, 1, permute, reverse)}))) &&
                positive(vertexMargins(base + ivec2(${cellOffset(0, 1, permute, reverse)})));
        }

        void main()
        {
            setOutput(float(cellRejected(outCoords())));
        }
        `
    }
}

/**
 * Binary OR of two 0/1 cell masks. Used to merge forward and reverse passes
 * into one bidirectional mask.
 */
class OrCellMasksProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(shape: Shape2)
    {
        this.outputShape = shape
        this.userCode = `
        uvec4 bits(vec4 v)
        {
            return uvec4(round(v)) & 1u;
        }

        ivec2 outCoords()
        {
            ivec2 p = getOutputCoords();
            return ivec2(p.y, p.x);
        }

        void main()
        {
            ivec2 p = outCoords();
            setOutput(vec4(bits(getA(p.y, p.x)) | bits(getB(p.y, p.x))));
        }
        `
    }
}

/**
 * Packs two ray classes for one dominant axis into the low two bits.
 */
class PackAxisProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(shape: Shape2)
    {
        this.outputShape = shape
        this.userCode = `
        uvec4 bit(vec4 v)
        {
            return uvec4(round(v)) & 1u;
        }

        ivec2 outCoords()
        {
            ivec2 p = getOutputCoords();
            return ivec2(p.y, p.x);
        }

        void main()
        {
            ivec2 p = outCoords();

            uvec4 a = bit(getA(p.y, p.x));
            uvec4 b = bit(getB(p.y, p.x));

            setOutput(vec4((a << 0u) | (b << 1u)));
        }
        `
    }
}

/**
 * Packs the x- and y-dominant axis groups into one 4-bit value per cell.
 */
class PackExtendedProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(shape: Shape2)
    {
        this.outputShape = shape
        this.userCode = `
        uvec4 pair(vec4 v)
        {
            return uvec4(round(v)) & 3u;
        }

        ivec2 outCoords()
        {
            ivec2 p = getOutputCoords();
            return ivec2(p.y, p.x);
        }

        void main()
        {
            ivec2 p = outCoords();

            uvec4 x = pair(getA(p.y, p.x));
            uvec4 y = pair(getB(p.y, p.x));

            setOutput(vec4((x << 0u) | (y << 2u)));
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

    constructor(shape: Shape2)
    {
        this.outputShape = shape
        this.userCode = `
        uvec4 pair(vec4 v)
        {
            return uvec4(round(v)) & 3u;
        }

        ivec2 outCoords()
        {
            ivec2 p = getOutputCoords();
            return ivec2(p.y, p.x);
        }

        void main()
        {
            ivec2 p = outCoords();
            uvec4 packed = pair(getA(p.y, p.x));
            setOutput(vec4((packed >> maskBit) & 1u));
        }
        `
    }
}

/**
 * Debug helper program: extracts one bit from the final 4-bit packed mask.
 */
class UnpackExtendedProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'maskBit', type: 'int' as const }]

    constructor(shape: Shape2)
    {
        this.outputShape = shape
        this.userCode = `
        uvec4 nibble(vec4 v)
        {
            return uvec4(round(v)) & 15u;
        }

        ivec2 outCoords()
        {
            ivec2 p = getOutputCoords();
            return ivec2(p.y, p.x);
        }

        void main()
        {
            ivec2 p = outCoords();
            uvec4 packed = nibble(getA(p.y, p.x));
            setOutput(vec4((packed >> maskBit) & 1u));
        }
        `
    }
}

/**
 * Runs the initial-margin pass and propagates margins through the image one
 * line at a time along the selected sweep axis.
 */
function buildMargins(
    image: tf.Tensor2D,
    permute: Permute2D,
    reverse: Reverse2D,
    verbose: boolean = false
): tf.Tensor4D
{
    const axis = permute[0]
    const backwards = reverse.includes(axis)

    const initialMarginsProgram = new InitialMarginsProgram(image.shape as Shape2, permute, reverse)
    let margins = runWebGLProgram(initialMarginsProgram, [image], 'float32', [], true) as tf.Tensor4D
    if (verbose) logMean('initialMargins2D', margins)

    const lines = tf.split(margins, margins.shape[axis], axis) as tf.Tensor4D[]
    tf.dispose(margins)

    const shape = lines[0].shape as PackedShape2
    const propagate = new PropagateMarginsProgram(shape, permute, reverse)

    const start = backwards ? lines.length - 2 : 1
    const end = backwards ? -1 : lines.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step)
    {
        const next = runWebGLProgram(
            propagate,
            [lines[i], lines[i - step]],
            'float32',
            [[i]],
            true
        ) as tf.Tensor4D

        tf.dispose(lines[i])
        lines[i] = next
    }

    margins = tf.concat(lines, axis) as tf.Tensor4D
    tf.dispose(lines)
    if (verbose) logMean('margins2D', margins)

    return margins
}

/**
 * Reverse-sweep version of buildMargins. It uses the forward-pass gates to
 * decide whether a route should extend only positive rejection margins or pass
 * signed margins through unchanged.
 */
function buildGatedMargins(
    image: tf.Tensor2D,
    gates: tf.Tensor4D,
    permute: Permute2D,
    reverse: Reverse2D,
    verbose: boolean = false
): tf.Tensor4D
{
    const axis = permute[0]
    const backwards = reverse.includes(axis)

    const initialMarginsProgram = new InitialMarginsProgram(image.shape as Shape2, permute, reverse)
    let margins = runWebGLProgram(initialMarginsProgram, [image], 'float32', [], true) as tf.Tensor4D
    if (verbose) logMean('initialMargins2D', margins)

    const lines = tf.split(margins, margins.shape[axis], axis) as tf.Tensor4D[]
    tf.dispose(margins)

    const shape = lines[0].shape as PackedShape2
    const propagate = new PropagateGatedMarginsProgram(shape, permute, reverse)

    const start = backwards ? lines.length - 2 : 1
    const end = backwards ? -1 : lines.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step)
    {
        const next = runWebGLProgram(
            propagate,
            [lines[i], lines[i - step], gates],
            'float32',
            [[i]],
            true
        ) as tf.Tensor4D

        tf.dispose(lines[i])
        lines[i] = next
    }

    margins = tf.concat(lines, axis) as tf.Tensor4D
    tf.dispose(lines)
    if (verbose) logMean('gatedMargins2D', margins)

    return margins
}

/**
 * Converts propagated vertex margins into a binary 2D cell mask.
 */
function classifyCellMask(
    margins: tf.Tensor4D,
    permute: Permute2D,
    reverse: Reverse2D,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor2D
{
    const shape = margins.shape.slice(0, 2) as Shape2
    const program = new ClassifyCellMaskProgram(shape, permute, reverse)

    const cellMask = runWebGLProgram(program, [margins], 'float32', [[tolerance]], true) as tf.Tensor2D
    if (verbose) logMean('cellMask2D', cellMask)

    return cellMask
}

/**
 * Creates the per-lane open/closed mask used by reverse propagation.
 */
function buildBackwardGates(
    forwardMask: tf.Tensor2D,
    permute: Permute2D,
    reverse: Reverse2D,
    verbose: boolean = false
): tf.Tensor4D
{
    const program = new BackwardGateProgram(forwardMask.shape as Shape2, permute, reverse)
    const gates = runWebGLProgram(program, [forwardMask], 'float32', [], true) as tf.Tensor4D

    if (verbose) logMean('gates2D', gates)

    return gates
}

/**
 * Merges forward and backward cell masks using a lane-wise binary OR.
 */
function orCellMasks(
    a: tf.Tensor2D,
    b: tf.Tensor2D,
    verbose: boolean = false
): tf.Tensor2D
{
    const program = new OrCellMasksProgram(a.shape as Shape2)
    const cellMask = runWebGLProgram(program, [a, b], 'float32', [], true) as tf.Tensor2D

    if (verbose) logMean('bidirectionalCellMask2D', cellMask)

    return cellMask
}

/**
 * Packs two directional cell masks belonging to one dominant axis.
 */
function packAxisMasks(
    masks: Pair<tf.Tensor2D>,
    verbose: boolean = false
): tf.Tensor2D
{
    const program = new PackAxisProgram(masks[0].shape as Shape2)
    const packed = runWebGLProgram(program, masks, 'float32', [], true) as tf.Tensor2D

    if (verbose) logMean('axisPack2D', packed)

    return packed
}

/**
 * Packs the x/y dominant-axis mask groups into the final 4-bit texture.
 */
function packExtendedMasks(
    masks: Pair<tf.Tensor2D>,
    verbose: boolean = false
): tf.Tensor2D
{
    const program = new PackExtendedProgram(masks[0].shape as Shape2)
    const packed = runWebGLProgram(program, masks, 'float32', [], true) as tf.Tensor2D

    if (verbose) logExtendedMasks(packed)

    return packed
}

function computeAxisPack(
    axis: Axis2D,
    computeMask: (quadrant: Quadrant) => tf.Tensor2D
): tf.Tensor2D
{
    const masks: tf.Tensor2D[] = []

    try
    {
        for (const quadrant of QUADRANTS[axis])
        {
            masks.push(computeMask(quadrant))
        }

        return packAxisMasks(masks as Pair<tf.Tensor2D>)
    }
    finally
    {
        tf.dispose(masks)
    }
}

function computeExtendedPack(
    computeMask: (axis: Axis2D) => tf.Tensor2D,
    verbose: boolean
): tf.Tensor2D
{
    const masks: tf.Tensor2D[] = []

    try
    {
        for (const axis of AXES)
        {
            masks.push(computeMask(axis))
        }

        return packExtendedMasks(masks as Pair<tf.Tensor2D>, verbose)
    }
    finally
    {
        tf.dispose(masks)
    }
}

/**
 * Computes one directed 2D cell mask for one dominant axis and quadrant.
 */
export function computeUnidirectionalShadowMap(
    image: tf.Tensor2D,
    dominantAxis: Axis2D,
    quadrant: Quadrant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor2D
{
    const { permute, reverse } = permuteReverseFromDominantAxisQuadrant(dominantAxis, quadrant)

    const margins = buildMargins(image, permute, reverse, verbose)
    const cellMask = classifyCellMask(margins, permute, reverse, tolerance, verbose)
    tf.dispose(margins)

    return cellMask
}

/**
 * Computes a 2D cell mask that considers both directions along the same line
 * family.
 */
export function computeBidirectionalShadowMap(
    image: tf.Tensor2D,
    dominantAxis: Axis2D,
    quadrant: Quadrant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor2D
{
    const { permute, reverse } = permuteReverseFromDominantAxisQuadrant(dominantAxis, quadrant)

    const forwardMask = computeUnidirectionalShadowMap(image, dominantAxis, quadrant, tolerance, verbose)
    const backwardReverse = complementReverse(reverse)
    const gates = buildBackwardGates(forwardMask, permute, backwardReverse, verbose)
    const backwardMargins = buildGatedMargins(image, gates, permute, backwardReverse, verbose)
    const backwardMask = classifyCellMask(backwardMargins, permute, backwardReverse, tolerance, verbose)
    const cellMask = orCellMasks(forwardMask, backwardMask, verbose)

    tf.dispose([gates, backwardMargins, forwardMask, backwardMask])

    return cellMask
}

/**
 * Convenience wrapper for querying the opposite quadrant.
 */
export function computeBidirectionalShadowMapReverse(
    image: tf.Tensor2D,
    dominantAxis: Axis2D,
    quadrant: Quadrant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor2D
{
    return computeBidirectionalShadowMap(image, dominantAxis, reverseQuadrant(quadrant), tolerance, verbose)
}

/**
 * Computes the 4-direction extended texture using forward-only cell masks.
 */
export function computeExtendedAnisotropicUnidirectionalShadowMap(
    image: tf.Tensor2D,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor2D
{
    return computeExtendedPack(
        (axis) => computeAxisPack(
            axis,
            (quadrant) => computeUnidirectionalShadowMap(image, axis, quadrant, tolerance)
        ),
        verbose
    )
}

/**
 * Computes the full 4-direction extended anisotropic bidirectional texture.
 */
export function computeExtendedAnisotropicBidirectionalShadowMap(
    image: tf.Tensor2D,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor2D
{
    return computeExtendedPack(
        (axis) => computeAxisPack(
            axis,
            (quadrant) => computeBidirectionalShadowMap(image, axis, quadrant, tolerance)
        ),
        verbose
    )
}

/**
 * Computes a bidirectional cell mask and reduces it into square blocks. Binary
 * min pooling marks a block rejected only if every covered cell-mask entry is
 * rejected.
 */
export function computeBidirectionalBlockShadowMap(
    image: tf.Tensor2D,
    dominantAxis: Axis2D,
    quadrant: Quadrant,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor2D
{
    const cellMask = computeBidirectionalShadowMap(image, dominantAxis, quadrant, tolerance, verbose)

    if (blockSize === 1) return cellMask

    const blocks = minPool2d(cellMask, blockSize, blockSize, 'same')
    if (verbose) logMean('blockCellMask2D', blocks)

    tf.dispose(cellMask)

    return blocks
}

/**
 * Computes the extended anisotropic bidirectional mask after block reduction of
 * each component mask.
 */
export function computeExtendedAnisotropicBidirectionalBlockShadowMap(
    image: tf.Tensor2D,
    tolerance: number,
    blockSize: number,
    verbose: boolean = false
): tf.Tensor2D
{
    return computeExtendedPack(
        (axis) => computeAxisPack(
            axis,
            (quadrant) => computeBidirectionalBlockShadowMap(image, axis, quadrant, tolerance, blockSize)
        ),
        verbose
    )
}

/**
 * Reference path for a unidirectional 2D cell mask.
 *
 * This physically transforms the image into canonical sweep orientation, runs
 * the same canonical implementation, then transforms the result back.
 */
export function computeUnidirectionalShadowMapReference(
    image: tf.Tensor2D,
    dominantAxis: Axis2D,
    quadrant: Quadrant,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor2D
{
    const { permute, reverse } = permuteReverseFromDominantAxisQuadrant(dominantAxis, quadrant)

    const reversed = image.reverse(reverse) as tf.Tensor2D
    const canonicalImage = reversed.transpose(permute) as tf.Tensor2D
    tf.dispose(reversed)

    const margins = buildMargins(canonicalImage, [0, 1], [], verbose)
    const canonicalMask = classifyCellMask(margins, [0, 1], [], tolerance, verbose)
    tf.dispose([margins, canonicalImage])

    const unpermuted = canonicalMask.transpose(inversePermutation(permute)) as tf.Tensor2D
    tf.dispose(canonicalMask)

    const cellMask = unpermuted.reverse(reverse) as tf.Tensor2D
    tf.dispose(unpermuted)

    return cellMask
}

/**
 * Reference path for the forward-only extended anisotropic 2D cell-mask
 * texture.
 */
export function computeExtendedAnisotropicUnidirectionalShadowMapReference(
    image: tf.Tensor2D,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor2D
{
    return computeExtendedPack(
        (axis) => computeAxisPack(
            axis,
            (quadrant) => computeUnidirectionalShadowMapReference(image, axis, quadrant, tolerance)
        ),
        verbose
    )
}

/**
 * Debug view that keeps one real y/++ mask and fills the rest with ones.
 */
export function computeExtendedAnisotropicBidirectionalShadowMapSingular(
    image: tf.Tensor2D,
    tolerance: number,
    verbose: boolean = false
): tf.Tensor2D
{
    const yMask = computeUnidirectionalShadowMap(image, 'y', '++', tolerance)
    const masks: tf.Tensor2D[] = []

    try
    {
        masks.push(packSyntheticAxis([tf.onesLike(yMask), tf.onesLike(yMask)]))
        masks.push(packSyntheticAxis([tf.clone(yMask), tf.onesLike(yMask)]))

        return packExtendedMasks(masks as Pair<tf.Tensor2D>, verbose)
    }
    finally
    {
        tf.dispose(masks)
        tf.dispose(yMask)
    }
}

function packSyntheticAxis(masks: Pair<tf.Tensor2D>): tf.Tensor2D
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

function minPool2d(
    input: tf.Tensor2D,
    filterSize: number,
    strides: number,
    pad: PadMode2D
): tf.Tensor2D
{
    return tf.tidy(() =>
    {
        const rank3 = input.expandDims(2) as tf.Tensor3D
        const pooled = tf.neg(tf.maxPool(tf.neg(rank3), [filterSize, filterSize], [strides, strides], pad))

        return pooled.squeeze([2]) as tf.Tensor2D
    })
}

/**
 * Logs the mean value over spatial axes. Useful for quick mask sanity checks
 * without downloading the whole texture.
 */
function logMean(name: string, tensor: tf.Tensor): void
{
    console.log(name, tf.tidy(() => tensor.mean([0, 1]).dataSync()))
}

function logAxisMasks(mask: tf.Tensor2D): void
{
    const unpack = new UnpackAxisProgram(mask.shape as Shape2)

    for (let maskBit = 0; maskBit < 2; maskBit++)
    {
        console.log(
            `cellMask2D${maskBit}`,
            tf.tidy(() => runWebGLProgram(unpack, [mask], 'float32', [[maskBit]]).mean([0, 1]).dataSync())
        )
    }
}

function logExtendedMasks(mask: tf.Tensor2D): void
{
    const unpack = new UnpackExtendedProgram(mask.shape as Shape2)
    const labels = [
        'cellMask2DX0',
        'cellMask2DX1',
        'cellMask2DY0',
        'cellMask2DY1',
    ]

    for (let maskBit = 0; maskBit < labels.length; maskBit++)
    {
        console.log(
            labels[maskBit],
            tf.tidy(() => runWebGLProgram(unpack, [mask], 'float32', [[maskBit]]).mean([0, 1]).dataSync())
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
