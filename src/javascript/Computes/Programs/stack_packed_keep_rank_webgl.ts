import * as tf from '@tensorflow/tfjs';
import { GPGPUProgram, useShapeUniforms } from '@tensorflow/tfjs-backend-webgl/dist/gpgpu_math';
import { getCoordsDataType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';

type Axis3 = 0 | 1 | 2;

function assertAxis3(axis: number): asserts axis is Axis3 
{
    
}

/**
 * Stack packed rank-5 slices over axis:
 * 
 * axis = 0 inputs: K tensors [1,H,W,2,2] -> output [K,H,W,2,2]
 * axis = 1 inputs: K tensors [D,1,W,2,2] -> output [D,K,W,2,2]
 * axis = 2 inputs: K tensors [D,H,1,2,2] -> output [D,H,K,2,2]
 *
 * Good for small K (<= 16) because it generates K samplers.
 */
export class StackSlicesPackedProgram implements GPGPUProgram 
{
    variableNames: string[]
    packedInputs = true
    packedOutput = true

    outputShape: number[]
    userCode: string
    enableShapeUniforms: boolean

    constructor(outputShape: number[], numSlices: number, axis: Axis3) 
    {
        this.outputShape = outputShape;
        this.enableShapeUniforms = useShapeUniforms(this.outputShape.length)

        this.variableNames = Array.from({ length: numSlices }, (_, i) => `S${i}`)

        const outRank = outputShape.length
        const dtype = getCoordsDataType(outRank)

        // Each input slice is rank-5 with a singleton on the stacked axis.
        // We always index that axis with 0.
        let kExpr: string
        let sliceCallArgs: string
        if (axis === 0) 
        {
            // input: [1, H, W, 2, 2] (outC.x is always 0)
            kExpr = 'd'
            sliceCallArgs = `0, h, w, r, c`
        } 
        else if (axis === 1) 
        {
            // input: [D, 1, W, 2, 2] (outC.y is always 0)
            kExpr = 'h' 
            sliceCallArgs = `d, 0, w, r, c`
        } 
        else
        {
            // input: [D, H, 1, 2, 2] (outC.z is always 0)
            kExpr = 'w'
            sliceCallArgs = `d, h, 0, r, c`
        }

        const selectSliceCode = () => 
        {
            const lines: string[] = []
            lines.push(`if (k == 0) { v = getS0(${sliceCallArgs}); }`)

            for (let i = 1; i < numSlices; i++) 
            lines.push(`else if (k == ${i}) { v = getS${i}(${sliceCallArgs}); }`)

            lines.push(`else { v = vec4(0.0); }`)
            return lines.join('\n        ')
        };

        this.userCode = `
        void main()
        {
            ${dtype} outC = getOutputCoords();
            int d = outC.x;
            int h = outC.y;
            int w = outC.z;
            int r = outC.w;
            int c = outC.u;

            int k = ${kExpr};

            vec4 v;
            ${selectSliceCode()}
            setOutput(v);
        }`
    }
}

/**
 * Stack packed rank-5 blocks over axis:
 * 
 * axis = 0 inputs: N block tensors [Di,H,W,2,2] -> output [sumDi,H,W,2,2]
 * axis = 1 inputs: N block tensors [D,Hi,W,2,2] -> output [D,sumHi,W,2,2]
 * axis = 2 inputs: N block tensors [D,H,Wi,2,2] -> output [D,H,sumWi,2,2]
 */
export class StackBlocksPackedProgram implements GPGPUProgram 
{
    variableNames: string[]
    packedInputs = true
    packedOutput = true

    outputShape: number[]
    userCode: string
    enableShapeUniforms: boolean

    constructor(outputShape: number[], blockSizes: number[], axis: Axis3) 
    {
        this.outputShape = outputShape
        this.enableShapeUniforms = useShapeUniforms(this.outputShape.length)

        const numBlocks = blockSizes.length
        this.variableNames = Array.from({ length: numBlocks }, (_, i) => `B${i}`)

        const outRank = outputShape.length // 5
        const dtype = getCoordsDataType(outRank)

        // Each input slice is rank-5 with a singleton on the stacked axis.
        // We always index that axis with 0.
        let kExpr: string
        let blockCallArgs: (string: any) => string
        if (axis === 0) 
        {
            // input: [localK, H, W, 2, 2] 
            kExpr = 'd'
            blockCallArgs = (localK: string) => `${localK}, h, w, r, c`
        } 
        else if (axis === 1) 
        {
            // input: [D, localK, W, 2, 2] 
            kExpr = 'h' 
            blockCallArgs = (localK: string) => `d, ${localK}, w, r, c`
        } 
        else
        {
            // input: [D, H, localK, 2, 2] 
            kExpr = 'w'
            blockCallArgs = (localK: string) => `d, h, ${localK}, r, c`
        }
   
        const thresholds: number[] = []
        let acc = 0
        for (const s of blockSizes) 
        {
            acc += s
            thresholds.push(acc)
        }

        const selectBlockCode = () => 
        {
            const lines: string[] = []
            let prev = 0;

            lines.push(`if (k < ${thresholds[0]}) { v = getB0(${blockCallArgs('k')}); }`)
            prev = thresholds[0]

            for (let i = 1; i < numBlocks; i++) 
            {
                const t = thresholds[i]
                const localK = `(k - ${prev})`
                lines.push(`else if (k < ${t}) { v = getB${i}(${blockCallArgs(localK)}); }`)
                prev = t
            }

            lines.push(`else { v = vec4(0.); }`);
            return lines.join('\n        ');
        }

        this.userCode = `
        void main()
        {
            ${dtype} outC = getOutputCoords();
            int d = outC.x;
            int h = outC.y;
            int w = outC.z;
            int r = outC.w;
            int c = outC.u;

            int k = ${kExpr};

            vec4 v;
            ${selectBlockCode()}
            setOutput(v);
        }`
    }
}

/**
 * Single-pass stack for KEEP-DIMS slices (UNBATCHED ONLY).
 * Inputs must be rank-5, with a singleton on the stacked axis:
 *  axis=0: [1,H,W,2,2], axis=1: [D,1,W,2,2], axis=2: [D,H,1,2,2]
 */
export function stackSlicesPacked(slices: tf.Tensor[], axis: Axis3 = 0): tf.Tensor 
{
    if (slices.length === 0) 
    {
        throw new Error('stackSlicesPacked: slices is empty.')
    }

    const K = slices.length

    if (slices[0].rank !== 5) 
    {
        throw new Error(`Expected slice rank 5. Got rank=${slices[0].rank}.`)
    }

    const firstShape = slices[0].shape.slice() // [*,*,*,2,2]

    if (firstShape.length !== 5) 
    {
        throw new Error(`slice[0] expected rank 5, got shape=${firstShape}.`)
    }

    if (firstShape[3] !== 2 || firstShape[4] !== 2) 
    {
        throw new Error(`Expected trailing [2,2], got [${firstShape[3]},${firstShape[4]}] on slice 0.`)
    }

    const dtype = slices[0].dtype

    for (let i = 0; i < K; i++) 
    {
        const s = slices[i]
        if (s.rank !== 5) 
        {
            throw new Error(`slice[${i}] expected rank 5, got rank=${s.rank}.`)
        }
        if (s.dtype !== dtype) 
        {
            throw new Error(`All slices must share dtype. slice[0]=${dtype}, slice[${i}]=${s.dtype}.`)
        }
        if (!s.shape.every((v, idx) => v === firstShape[idx])) 
        {
            throw new Error(`All slices must share shape. slice[0]=${firstShape}, slice[${i}]=${s.shape}.`)
        }
    }

    let D: number, H: number, W: number
    if (axis === 0) 
    { 
        [D, H, W] = [K, firstShape[1], firstShape[2]]
    }
    else if (axis === 1) 
    { 
        [D, H, W] = [firstShape[0], K, firstShape[2]]
    }
    else 
    { 
        [D, H, W] = [firstShape[0], firstShape[1], K]
    }

    const outShape = [D, H, W, 2, 2]

    const backend: any = tf.backend()
    const prog = new StackSlicesPackedProgram(outShape, K, axis)

    const info = backend.runWebGLProgram(prog, slices, dtype, [], true)
    return tf.engine().makeTensorFromTensorInfo(info)
}


/**
 * Single-pass concat of a small number of axis-blocks (UNBATCHED ONLY).
 */
export function stackBlocksPacked(blocks: tf.Tensor[], axis: Axis3 = 0): tf.Tensor 
{
    if (blocks.length === 0) 
    {
        throw new Error('stackBlocksPacked: blocks is empty.')
    }

    assertAxis3(axis)

    const N = blocks.length
    if (blocks[0].rank !== 5) 
    {
        throw new Error(`Expected block rank 5 ([D,H,W,2,2]). Got rank=${blocks[0].rank}.`)
    }

    const dtype = blocks[0].dtype
    const base = blocks[0].shape.slice() // [D,H,W,2,2]

    if (base[3] !== 2 || base[4] !== 2) 
    {
        throw new Error(`Expected trailing [2,2], got [${base[3]},${base[4]}] on block 0.`)
    }

    const [baseD, baseH, baseW] = base

    const blockSizes: number[] = []
    for (let i = 0; i < N; i++) 
    {
        const t = blocks[i];
        if (t.rank !== 5) throw new Error(`block[${i}] expected rank 5, got rank=${t.rank}.`);
        if (t.dtype !== dtype) throw new Error(`All blocks must share dtype.`);
        const s = t.shape;
        if (s[3] !== 2 || s[4] !== 2) throw new Error(`Expected trailing [2,2] on block[${i}].`);

        if (axis === 0) {
        if (s[1] !== baseH || s[2] !== baseW) throw new Error(`Inconsistent block[${i}] shape for axis=0.`);
        } else if (axis === 1) {
        if (s[0] !== baseD || s[2] !== baseW) throw new Error(`Inconsistent block[${i}] shape for axis=1.`);
        } else {
        if (s[0] !== baseD || s[1] !== baseH) throw new Error(`Inconsistent block[${i}] shape for axis=2.`);
        }

        blockSizes.push(s[axis]);
    }

    const total = blockSizes.reduce((a, b) => a + b, 0);

    let outShape: [number, number, number, 2, 2]
    if (axis === 0) 
    { 
        outShape = [total, baseH, baseW, 2, 2]
    }
    else if (axis === 1) 
    { 
        outShape = [baseD, total, baseW, 2, 2]
    }
    else 
    { 
        outShape = [baseD, baseH, total, 2, 2]
    }

    const backend: any = tf.backend()
    const prog = new StackBlocksPackedProgram(outShape, blockSizes, axis)

    const info = backend.runWebGLProgram(prog, blocks, dtype, [], true)
    return tf.engine().makeTensorFromTensorInfo(info)
}

/**
 * Multi-pass stack for slices:
 * 
 * axis = 0: slices [1,H,W,2,2] -> out [D,H,W,2,2]
 * axis = 1: slices [D,1,W,2,2] -> out [D,H,W,2,2]
 * axis = 2: slices [D,H,1,2,2] -> out [D,H,W,2,2]
 */
export function stackPacked(slices: tf.Tensor[], axis: Axis3 = 0, chunkSize?: number): tf.Tensor 
{
    if (slices.length === 0) 
    {
        throw new Error('stackPacked: slices is empty.')
    }

    if (axis !== 0 && axis !== 1 && axis !== 2) 
    {
        throw new Error(`axis must be 0, 1, or 2. Got axis=${axis}.`);
    }

    const K = chunkSize ?? 16

    if (slices.length <= K) 
    {
        return stackSlicesPacked(slices, axis)
    }

    let level: tf.Tensor[] = []
    for (let i = 0; i < slices.length; i += K) 
    {
        const chunk = slices.slice(i, i + K)
        level.push(stackSlicesPacked(chunk, axis))
    }

    while (level.length > 1) 
    {
        const next: tf.Tensor[] = []
        for (let i = 0; i < level.length; i += K) 
        {
            next.push(stackBlocksPacked(level.slice(i, i + K), axis))
        }

        for (const t of level) t.dispose()
        level = next
    }

    return level[0]
}
