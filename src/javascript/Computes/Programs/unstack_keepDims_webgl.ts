import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram, useShapeUniforms } from '@tensorflow/tfjs-backend-webgl/dist/gpgpu_math'
import { getCoordsDataType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler'

type Axis3 = 0 | 1 | 2

/**
 * Unstack unpacked rank-3 tensor over axis keeping rank:
 * input:  [D, H, W]
 *
 * axis = 0 (depth):  returns D tensors, each [1, H, W]
 * axis = 1 (height): returns H tensors, each [D, 1, W]
 * axis = 2 (width):  returns W tensors, each [D, H, 1]
 */
export class Unstack3dProgram implements GPGPUProgram 
{
    variableNames = ['A']
    packedInputs = false
    packedOutput = false

    customUniforms = [{ name: 'uSlice', type: 'int' as const }]

    outputShape: number[]
    userCode: string
    enableShapeUniforms: boolean

    constructor(outputShape: number[], axis: Axis3)
    {
        // outputShape must be rank-3: [D', H', W'] where one of D'|H'|W' is 1
        this.outputShape = outputShape
        this.enableShapeUniforms = useShapeUniforms(this.outputShape.length)

        const outRank = outputShape.length // should be 3
        const dtype = getCoordsDataType(outRank)

        // For rank-3, tf coords are typically outC.x, outC.y, outC.z
        // representing [d, h, w] in our output layout.
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

            ${mapCoords}

            float v = getA(d, h, w);
            setOutput(v);
        }
        `
    }
}

export function unstack3d(tensor: tf.Tensor, axis: Axis3 = 0): tf.Tensor[]
{
    assertAxis3(axis)
    assertTensor3d(tensor)

    const [D, H, W] = tensor.shape as [number, number, number]
    const sliceCount = [D, H, W][axis]

    // Keep the chosen axis as 1 instead of removing it.
    const outBase: [number, number, number] = [D, H, W]
    outBase[axis] = 1

    const outShape = [...outBase] // rank-3

    const backend: any = tf.backend()
    const prog = new Unstack3dProgram(outShape, axis)

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

function assertTensor3d(tensor: tf.Tensor): void
{
    if (tensor.rank !== 3) 
    {
        throw new Error(`Expected rank-3 [D,H,W]. Got rank=${tensor.rank}.`)
    }
}
