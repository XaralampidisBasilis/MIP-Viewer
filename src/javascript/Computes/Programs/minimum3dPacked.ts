import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

type Shape3Packed = [number, number, number, 2, 2]


function validatePackedTensor5d(tensor: tf.Tensor, name: string): void
{
    if (tensor.rank !== 5)
    {
        throw new Error(`${name} must be rank 5 [D, H, W, 2, 2]. Got rank=${tensor.rank}.`)
    }

    const shape = tensor.shape as number[]
    if (shape[3] !== 2 || shape[4] !== 2)
    {
        throw new Error(`${name} must have trailing [2, 2]. Got shape=[${shape.join(', ')}].`)
    }
}

function sameShape(a: readonly number[], b: readonly number[]): boolean
{
    return a.length === b.length && a.every((size, i) => size === b[i])
}

function runWebGLProgram(
    program: GPGPUProgram,
    inputs: tf.Tensor[],
    dtype: tf.DataType,
): tf.Tensor
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(program, inputs, dtype, [], true)
    return tf.engine().makeTensorFromTensorInfo(info)
}


export class Minimum3dPackedProgram implements GPGPUProgram
{
    variableNames = ['A', 'B']
    outputShape: Shape3Packed
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(outputShape: Shape3Packed)
    {
        this.outputShape = outputShape
        this.userCode = `
        void main()
        {
            vec4 a = getAAtOutCoords();
            vec4 b = getBAtOutCoords();

            setOutput(min(a, b));
        }
        `
    }
}

/**
 * Packed rank-5 variant of tf.minimum for tensors shaped [D, H, W, 2, 2].
 * The output keeps the same packed layout.
 */
export function minimum3dPacked(
    a: tf.Tensor5D,
    b: tf.Tensor5D,
): tf.Tensor5D
{
    validatePackedTensor5d(a, 'a')
    validatePackedTensor5d(b, 'b')

    if (!sameShape(a.shape, b.shape))
    {
        throw new Error(`a and b must have the same shape. a=[${a.shape.join(', ')}], b=[${b.shape.join(', ')}].`)
    }

    if (a.dtype !== b.dtype)
    {
        throw new Error(`a and b must have the same dtype. a=${a.dtype}, b=${b.dtype}.`)
    }

    const program = new Minimum3dPackedProgram(a.shape as Shape3Packed)
    return runWebGLProgram(program, [a, b], a.dtype) as tf.Tensor5D
}
