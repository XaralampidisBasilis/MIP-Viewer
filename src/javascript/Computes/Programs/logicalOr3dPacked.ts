import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

type Shape3Packed = [number, number, number, 2, 2]


function validatePackedBoolTensor5d(tensor: tf.Tensor, name: string): void
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

    if (tensor.dtype !== 'bool')
    {
        throw new Error(`${name} must have dtype bool. Got dtype=${tensor.dtype}.`)
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

export class LogicalOr3dPackedProgram implements GPGPUProgram
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

            setOutput(vec4(greaterThan(max(a, b), vec4(0.5))));
        }
        `
    }
}

/**
 * Packed rank-5 variant of tf.logicalOr for bool tensors shaped [D, H, W, 2, 2].
 * The output keeps the same packed layout.
 */
export function logicalOr3dPacked(
    a: tf.Tensor5D,
    b: tf.Tensor5D,
): tf.Tensor5D
{
    validatePackedBoolTensor5d(a, 'a')
    validatePackedBoolTensor5d(b, 'b')

    if (!sameShape(a.shape, b.shape))
    {
        throw new Error(`a and b must have the same shape. a=[${a.shape.join(', ')}], b=[${b.shape.join(', ')}].`)
    }

    const program = new LogicalOr3dPackedProgram(a.shape as Shape3Packed)
    return runWebGLProgram(program, [a, b], 'bool') as tf.Tensor5D
}
