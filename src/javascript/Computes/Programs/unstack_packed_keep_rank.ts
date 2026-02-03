import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram, useShapeUniforms } from '@tensorflow/tfjs-backend-webgl/dist/gpgpu_math'
import { getCoordsDataType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler'

type Axis3 = 0 | 1 | 2

/**
 * Unstack packed rank-5 tensor over axis keeping rank:
 * input:  [D, H, W, 2, 2]
 *
 * axis = 0 (depth):  returns D tensors, each [1, H, W, 2, 2]
 * axis = 1 (height): returns H tensors, each [D, 1, W, 2, 2]
 * axis = 2 (width):  returns W tensors, each [D, H, 1, 2, 2]
 */
export class UnstackPackedProgram implements GPGPUProgram 
{
    variableNames = ['A']
    packedInputs = true
    packedOutput = true

    customUniforms = [{ name: 'uSlice', type: 'int' as const }]

    outputShape: number[]
    userCode: string
    enableShapeUniforms: boolean

    constructor(outputShape: number[], axis: Axis3) 
    {
        // outputShape must be rank-5: [D', H', W', 2, 2] where one of D'|H'|W' is 1
        this.outputShape = outputShape
        this.enableShapeUniforms = useShapeUniforms(this.outputShape.length)

        const outRank = outputShape.length // should be 5
        const dtype = getCoordsDataType(outRank)

        // For rank-5, tf coords are typically outC.x, outC.y, outC.z, outC.w, outC.u
        // representing [d, h, w, r, c] in our output layout.
        let mapCoords: string
        if (axis === 0) 
        {
            // output: [1, H, W, 2, 2] (outC.x is always 0)
            mapCoords = `
            int d = uSlice;
            int h = outC.y;
            int w = outC.z;
            `
        } 
        else if (axis === 1) 
        {
            // output: [D, 1, W, 2, 2] (outC.y is always 0)
            mapCoords = `
            int d = outC.x;
            int h = uSlice;
            int w = outC.z;
            `
        } 
        else
        {
            // axis === 2
            // output: [D, H, 1, 2, 2] (outC.z is always 0)
            mapCoords = `
            int d = outC.x;
            int h = outC.y;
            int w = uSlice;
            `
        }

        this.userCode = `
        void main() 
        {
            ${dtype} outC = getOutputCoords();

            int r = outC.w;
            int c = outC.u;

            ${mapCoords}

            vec4 v = getA(d, h, w, r, c);
            setOutput(v);
        }
        `
    }
}

export function unstackPacked(T: tf.Tensor, axis: Axis3 = 0): tf.Tensor[]
{
    if (T.rank !== 5) 
    {
        throw new Error(`Unstack expects rank-5 [D,H,W,2,2]. Got rank=${T.rank}.`)
    }
    
    if (axis !== 0 && axis !== 1 && axis !== 2) 
    {
        throw new Error(`Axis must be 0, 1, or 2. Got axis=${axis}.`)
    }
    
    const [D, H, W, twoR, twoC] = T.shape as unknown as [number, number, number, number, number]

    if (twoR !== 2 || twoC !== 2) 
    {
        throw new Error(`Expected trailing [2,2], got [${twoR},${twoC}].`)
    }
    
    const sliceCount = [D, H, W][axis]

    // Keep the chosen axis as 1 instead of removing it.
    const outBase3: [number, number, number] = [D, H, W]
    outBase3[axis] = 1

    const outShape = [...outBase3, 2, 2] // rank-5

    const backend: any = tf.backend()
    const prog = new UnstackPackedProgram(outShape, axis)

    const ys: tf.Tensor[] = []
    for (let i = 0; i < sliceCount; i++) 
    {
        // uSlice uniform as 1-element array
        const info = backend.runWebGLProgram(prog, [T], T.dtype, [[i]], true)
        ys.push(tf.engine().makeTensorFromTensorInfo(info))
    }

    return ys
}
