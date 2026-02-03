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
        const outCoord = ['outC.x', 'outC.y', 'outC.z']
        outCoord[axis] = 'uSlice'

        const mapCoords = `
        int d = ${outCoord[0]};
        int h = ${outCoord[1]};
        int w = ${outCoord[2]};
        `
        
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

export function unstackPacked(tensor: tf.Tensor, axis: Axis3 = 0): tf.Tensor[]
{
    assertAxis3(axis)
    assertTensor(tensor)
    
    const [D, H, W] = tensor.shape as unknown as [number, number, number, number, number]
    const sliceCount = [D, H, W][axis]

    // Keep the chosen axis as 1 instead of removing it.
    const outBase: [number, number, number] = [D, H, W]
    outBase[axis] = 1

    const outShape = [...outBase, 2, 2] // rank-5

    const backend: any = tf.backend()
    const prog = new UnstackPackedProgram(outShape, axis)

    const ys: tf.Tensor[] = []
    for (let i = 0; i < sliceCount; i++) 
    {
        const info = backend.runWebGLProgram(prog, [tensor], tensor.dtype, [[i]], true)
        ys.push(tf.engine().makeTensorFromTensorInfo(info))
    }

    return ys
}

// assertions

function assertAxis3(axis: number): asserts axis is Axis3 
{
    if (axis !== 0 && axis !== 1 && axis !== 2) 
    {
        throw new Error(`axis must be 0, 1, or 2. Got axis=${axis}.`);
    }
}

function assertTensor(tensor: tf.Tensor): void
{
    if (tensor.rank !== 5) 
    {
        throw new Error(`Expected rank-5 [D,H,W,2,2]. Got rank=${tensor.rank}.`)
    }

    const shape = tensor.shape as unknown as [number, number, number, number, number]
    const R = shape[3]
    const C = shape[4]

    if (R !== 2 || C !== 2) 
    {
        throw new Error(`Expected trailing [2,2], got [${R},${C}].`)
    }
}
