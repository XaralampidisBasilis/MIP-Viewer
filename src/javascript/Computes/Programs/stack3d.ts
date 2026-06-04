import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram, useShapeUniforms } from '@tensorflow/tfjs-backend-webgl/dist/gpgpu_math'
import { getCoordsDataType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler'

type Axis3 = 0 | 1 | 2

/**
 * Stack rank-3 slices over axis:
 * 
 * axis = 0 inputs: K tensors [1,H,W] -> output [K,H,W]
 * axis = 1 inputs: K tensors [D,1,W] -> output [D,K,W]
 * axis = 2 inputs: K tensors [D,H,1] -> output [D,H,K]
 *
 * Good for small K (<= 16) because it generates K samplers.
 */
export class StackSlices3dProgram implements GPGPUProgram
{
    variableNames: string[]
    packedInputs = false
    packedOutput = false

    outputShape: number[]
    userCode: string
    enableShapeUniforms: boolean

    constructor(outputShape: number[], numSlices: number, axis: Axis3)
    {
        this.outputShape = outputShape
        this.enableShapeUniforms = useShapeUniforms(this.outputShape.length)

        this.variableNames = Array.from({ length: numSlices }, (_, i) => `S${i}`)

        const outRank = outputShape.length
        const dtype = getCoordsDataType(outRank)

        // Each input slice is rank-3 with a singleton on the stacked axis.
        // We always index that axis with 0.
        const inCoord = ['d', 'h', 'w']
        inCoord[axis] = '0'

        const sliceCallArgs = `${inCoord[0]}, ${inCoord[1]}, ${inCoord[2]}`
        const kExpr = ['d', 'h', 'w'][axis]

        const selectSliceCode = () =>
        {
            const lines: string[] = []
            lines.push(`if (k == 0) { v = getS0(${sliceCallArgs}); }`)

            for (let i = 1; i < numSlices; i++)
            lines.push(`else if (k == ${i}) { v = getS${i}(${sliceCallArgs}); }`)

            lines.push(`else { v = 0.0; }`)
            return lines.join('\n        ')
        }

        this.userCode = `
        void main()
        {
            ${dtype} outC = getOutputCoords();
            int d = outC.x;
            int h = outC.y;
            int w = outC.z;

            int k = ${kExpr};

            float v;
            ${selectSliceCode()}
            setOutput(v);
        }`
    }
}

/**
 * Stack rank-3 blocks over axis:
 * 
 * axis = 0 inputs: N block tensors [Di,H,W] -> output [sumDi,H,W]
 * axis = 1 inputs: N block tensors [D,Hi,W] -> output [D,sumHi,W]
 * axis = 2 inputs: N block tensors [D,H,Wi] -> output [D,H,sumWi]
 */
export class StackBlocks3dProgram implements GPGPUProgram
{
    variableNames: string[]
    packedInputs = false
    packedOutput = false

    outputShape: number[]
    userCode: string
    enableShapeUniforms: boolean

    constructor(outputShape: number[], blockSizes: number[], axis: Axis3)
    {
        this.outputShape = outputShape
        this.enableShapeUniforms = useShapeUniforms(this.outputShape.length)

        const numBlocks = blockSizes.length
        this.variableNames = Array.from({ length: numBlocks }, (_, i) => `B${i}`)

        const outRank = outputShape.length // 3
        const dtype = getCoordsDataType(outRank)

        const blockCallArgs = (localK: string) =>
        {
            const args = ['d', 'h', 'w']
            args[axis] = localK

            return `${args[0]}, ${args[1]}, ${args[2]}`
        }

        let acc = 0
        const kEnds = blockSizes.map(s => (acc += s))
        const kExpr = ['d', 'h', 'w'][axis]

        const selectBlockCode = () =>
        {
            const lines: string[] = []

            lines.push(`if (k < ${kEnds[0]}) { v = getB0(${blockCallArgs('k')}); }`)

            for (let i = 1; i < numBlocks; i++)
            {
                const localK = `(k - ${kEnds[i - 1]})`
                lines.push(`else if (k < ${kEnds[i]}) { v = getB${i}(${blockCallArgs(localK)}); }`)
            }

            lines.push(`else { v = 0.0; }`)
            return lines.join('\n        ')
        }

        this.userCode = `
        void main()
        {
            ${dtype} outC = getOutputCoords();
            int d = outC.x;
            int h = outC.y;
            int w = outC.z;

            int k = ${kExpr};

            float v;
            ${selectBlockCode()}
            setOutput(v);
        }`
    }
}

/**
 * Single-pass stack for KEEP-DIMS slices (UNBATCHED ONLY).
 * Inputs must be rank-3, with a singleton on the stacked axis:
 *  axis=0: [1,H,W], axis=1: [D,1,W], axis=2: [D,H,1]
 */
export function stackSlices3d(slices: tf.Tensor[], axis: Axis3 = 0): tf.Tensor
{
    assertAxis3(axis)
    assertSlices3d(slices, axis)

    const K = slices.length
    const dtype = slices[0].dtype
    const shape = slices[0].shape as [number, number, number]

    const outBase = [shape[0], shape[1], shape[2]]
    outBase[axis] = K

    const outShape = [outBase[0], outBase[1], outBase[2]]

    const backend: any = tf.backend()
    const prog = new StackSlices3dProgram(outShape, K, axis)

    const info = backend.runWebGLProgram(prog, slices, dtype, [], true)
    return tf.engine().makeTensorFromTensorInfo(info)
}

/**
 * Single-pass concat of a small number of axis-blocks.
 */
export function stackBlocks3d(blocks: tf.Tensor[], axis: Axis3 = 0): tf.Tensor
{
    assertAxis3(axis)
    assertBlocks3d(blocks, axis)

    const N = blocks.length
    const dtype = blocks[0].dtype
    const shape = blocks[0].shape as [number, number, number]

    const blockSizes: number[] = []
    for (let i = 0; i < N; i++)
    {
        blockSizes.push(blocks[i].shape[axis] as number)
    }

    const total = blockSizes.reduce((a, b) => a + b, 0)

    const outBase = [shape[0], shape[1], shape[2]]
    outBase[axis] = total

    const outShape = [outBase[0], outBase[1], outBase[2]]

    const backend: any = tf.backend()
    const prog = new StackBlocks3dProgram(outShape, blockSizes, axis)

    const info = backend.runWebGLProgram(prog, blocks, dtype, [], true)
    return tf.engine().makeTensorFromTensorInfo(info)
}

/**
 * Multi-pass stack for slices:
 * 
 * axis = 0: slices [1,H,W] -> out [D,H,W]
 * axis = 1: slices [D,1,W] -> out [D,H,W]
 * axis = 2: slices [D,H,1] -> out [D,H,W]
 */
export function stack3d(slices: tf.Tensor[], axis: Axis3 = 0, chunkSize?: number): tf.Tensor
{
    assertAxis3(axis)
    assertSlices3d(slices, axis)

    const K = getChunkSize(chunkSize)

    // Fast path: one pass, K samplers total.
    if (slices.length <= K)
    {
        return stackSlices3d(slices, axis)
    }

    // Pass 1: stack slices in groups of K -> produces "blocks" along axis.
    let level: tf.Tensor[] = []
    for (let start = 0; start < slices.length; start += K)
    {
        const end = Math.min(start + K, slices.length)
        const chunk = slices.slice(start, end)
        level.push(stackSlices3d(chunk, axis))
    }

    // Reduction passes: repeatedly concat blocks in groups of K until one remains.
    while (level.length > 1)
    {
        const next: tf.Tensor[] = []

        for (let start = 0; start < level.length; start += K)
        {
            const end = Math.min(start + K, level.length)
            const group = level.slice(start, end)
            next.push(stackBlocks3d(group, axis))
        }

        for (const t of level) t.dispose()
        level = next
    }

    return level[0]
}

// assertions

function assertAxis3(axis: number): asserts axis is Axis3
{
    if (axis !== 0 && axis !== 1 && axis !== 2)
    {
        throw new Error(`axis must be 0, 1, or 2. Got axis=${axis}.`)
    }
}

function assertSlices3d(slices: tf.Tensor[], axis: Axis3 = 0): void
{
    if (slices.length === 0)
    {
        throw new Error('slices is empty.')
    }

    const firstSlice = slices[0]

    if (firstSlice.rank !== 3)
    {
        throw new Error(`Expected slice rank 3 ([D,H,W]). Got rank=${firstSlice.rank}.`)
    }

    const dtype = firstSlice.dtype
    const base = firstSlice.shape.slice() // [D,H,W]

    if (base.length !== 3)
    {
        throw new Error(`slice[0] expected rank 3, got shape=${base}.`)
    }

    // Slices must have singleton on the stacked axis.
    if (base[axis] !== 1)
    {
        throw new Error(`Expected singleton (==1) on axis=${axis} for slice 0, got ${base[axis]}.`)
    }

    const K = slices.length
    const [baseD, baseH, baseW] = base

    for (let i = 0; i < K; i++)
    {
        const s = slices[i]

        if (s.rank !== 3)
        {
            throw new Error(`slice[${i}] expected rank 3, got rank=${s.rank}.`)
        }
        if (s.dtype !== dtype)
        {
            throw new Error(`All slices must share dtype. slice[0]=${dtype}, slice[${i}]=${s.dtype}.`)
        }

        const sh = s.shape

        if (sh.length !== 3)
        {
            throw new Error(`slice[${i}] expected rank 3, got shape=${sh}.`)
        }

        if (sh[axis] !== 1)
        {
            throw new Error(`Expected singleton (==1) on axis=${axis} for slice[${i}], got ${sh[axis]}.`)
        }

        if (axis === 0)
        {
            // expected [1, H, W]
            if (sh[1] !== baseH || sh[2] !== baseW)
            {
                throw new Error(`Inconsistent slice[${i}] shape for axis=0. base=[${baseD},${baseH},${baseW}], got=[${sh[0]},${sh[1]},${sh[2]}].`)
            }
        }
        else if (axis === 1)
        {
            // expected [D, 1, W]
            if (sh[0] !== baseD || sh[2] !== baseW)
            {
                throw new Error(`Inconsistent slice[${i}] shape for axis=1. base=[${baseD},${baseH},${baseW}], got=[${sh[0]},${sh[1]},${sh[2]}].`)
            }
        }
        else
        {
            // expected [D, H, 1]
            if (sh[0] !== baseD || sh[1] !== baseH)
            {
                throw new Error(`Inconsistent slice[${i}] shape for axis=2. base=[${baseD},${baseH},${baseW}], got=[${sh[0]},${sh[1]},${sh[2]}].`)
            }
        }
    }
}

function assertBlocks3d(blocks: tf.Tensor[], axis: Axis3 = 0): void
{
    if (blocks.length === 0)
    {
        throw new Error('blocks is empty.')
    }

    const firstBlock = blocks[0]

    if (firstBlock.rank !== 3)
    {
        throw new Error(`Expected block rank 3 ([D,H,W]). Got rank=${firstBlock.rank}.`)
    }

    const dtype = firstBlock.dtype
    const base = firstBlock.shape.slice() // [D,H,W]

    if (base.length !== 3)
    {
        throw new Error(`block[0] expected rank 3, got shape=${base}.`)
    }

    const N = blocks.length
    const [baseD, baseH, baseW] = base

    for (let i = 0; i < N; i++)
    {
        const b = blocks[i]

        if (b.rank !== 3)
        {
            throw new Error(`block[${i}] expected rank 3, got rank=${b.rank}.`)
        }
        if (b.dtype !== dtype)
        {
            throw new Error(`All blocks must share dtype. block[0]=${dtype}, block[${i}]=${b.dtype}.`)
        }

        const s = b.shape

        if (s.length !== 3)
        {
            throw new Error(`block[${i}] expected rank 3, got shape=${s}.`)
        }

        if (axis === 0)
        {
            if (s[1] !== baseH || s[2] !== baseW)
            {
                throw new Error(`Inconsistent block[${i}] shape for axis=0. base=[${baseD},${baseH},${baseW}], got=[${s[0]},${s[1]},${s[2]}].`)
            }
        }
        else if (axis === 1)
        {
            if (s[0] !== baseD || s[2] !== baseW)
            {
                throw new Error(`Inconsistent block[${i}] shape for axis=1. base=[${baseD},${baseH},${baseW}], got=[${s[0]},${s[1]},${s[2]}].`)
            }
        }
        else
        {
            if (s[0] !== baseD || s[1] !== baseH)
            {
                throw new Error(`Inconsistent block[${i}] shape for axis=2. base=[${baseD},${baseH},${baseW}], got=[${s[0]},${s[1]},${s[2]}].`)
            }
        }
    }
}

// helpers

function getDefaultChunkSize(): number
{
    const FALLBACK = 16
    const SAFETY = 4
    const MIN = 4
    const MAX = 16

    if (tf.getBackend() !== 'webgl')
    {
        return FALLBACK
    }

    const backend: any = tf.backend()
    const gpgpu = backend.getGPGPUContext?.() ?? backend.gpgpu
    const gl: WebGLRenderingContext | WebGL2RenderingContext | undefined = gpgpu?.gl

    if (!gl)
    {
        return FALLBACK
    }

    const maxUnits = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) as number
    return Math.max(MIN, Math.min(MAX, maxUnits - SAFETY))
}

function getChunkSize(chunkSize?: number): number
{
    const fallback = getDefaultChunkSize()

    if (chunkSize === undefined)
    {
        return fallback
    }

    const size = Math.floor(chunkSize)

    if (!Number.isFinite(size) || size < 1)
    {
        return fallback
    }

    return size
}
