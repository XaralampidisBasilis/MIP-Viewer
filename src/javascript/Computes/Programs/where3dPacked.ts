import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

type Shape3Packed = [number, number, number, 2, 2]
type WhereValue = tf.Tensor5D | number

export class Where3dPackedProgram implements GPGPUProgram
{
    variableNames: string[]
    outputShape: Shape3Packed
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms: { name: string, type: 'float' }[]

    constructor(
        outputShape: Shape3Packed,
        xIsScalar: boolean,
        yIsScalar: boolean,
    ) {
        this.outputShape = outputShape
        this.variableNames = ['Condition']
        this.customUniforms = []

        if (xIsScalar) this.customUniforms.push({ name: 'xValue', type: 'float' })
        else this.variableNames.push('X')

        if (yIsScalar) this.customUniforms.push({ name: 'yValue', type: 'float' })
        else this.variableNames.push('Y')

        const xExpr = xIsScalar ? 'vec4(xValue)' : 'getX(d, h, w, r, c)'
        const yExpr = yIsScalar ? 'vec4(yValue)' : 'getY(d, h, w, r, c)'

        this.userCode = `
        void main()
        {
            ivec5 coords = getOutputCoords();
            int d = coords.x;
            int h = coords.y;
            int w = coords.z;
            int r = coords.w;
            int c = coords.u;

            bvec4 condition = greaterThan(getCondition(d, h, w, r, c), vec4(0.5));
            vec4 x = ${xExpr};
            vec4 y = ${yExpr};

            setOutput(vec4(
                condition.r ? x.r : y.r,
                condition.g ? x.g : y.g,
                condition.b ? x.b : y.b,
                condition.a ? x.a : y.a
            ));
        }
        `
    }
}

/**
 * Packed rank-5 variant of tf.where for tensors shaped [D, H, W, 2, 2].
 * The condition and any tensor branches must have the same packed shape.
 */
export function where3dPacked(
    condition: tf.Tensor5D,
    x: WhereValue,
    y: WhereValue,
): tf.Tensor5D
{
    validatePackedTensor5d(condition, 'condition')

    const xIsScalar = isScalar(x)
    const yIsScalar = isScalar(y)

    if (!xIsScalar) validateBranchTensor(x, condition, 'x')
    if (!yIsScalar) validateBranchTensor(y, condition, 'y')

    const dtype = inferOutputDtype(x, y)
    const program = new Where3dPackedProgram(condition.shape as Shape3Packed, xIsScalar, yIsScalar)

    const inputs: tf.Tensor[] = [condition]
    const uniforms: number[][] = []

    if (xIsScalar) uniforms.push([x])
    else inputs.push(x)

    if (yIsScalar) uniforms.push([y])
    else inputs.push(y)

    return runWebGLProgram(program, inputs, dtype, uniforms) as tf.Tensor5D
}

function isScalar(value: WhereValue): value is number
{
    return typeof value === 'number'
}

function validateBranchTensor(
    tensor: tf.Tensor5D,
    condition: tf.Tensor5D,
    name: string,
): void
{
    validatePackedTensor5d(tensor, name)

    if (!sameShape(tensor.shape, condition.shape))
    {
        throw new Error(`${name} shape must match condition shape. condition=[${condition.shape.join(', ')}], ${name}=[${tensor.shape.join(', ')}].`)
    }
}

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

function inferOutputDtype(x: WhereValue, y: WhereValue): tf.DataType
{
    if (!isScalar(x) && !isScalar(y) && x.dtype !== y.dtype)
    {
        throw new Error(`x and y must have the same dtype. x=${x.dtype}, y=${y.dtype}.`)
    }

    if (!isScalar(x)) return x.dtype
    if (!isScalar(y)) return y.dtype

    return 'float32'
}

function runWebGLProgram(
    program: GPGPUProgram,
    inputs: tf.Tensor[],
    dtype: tf.DataType,
    uniforms: number[][],
): tf.Tensor
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(program, inputs, dtype, uniforms, true)
    return tf.engine().makeTensorFromTensorInfo(info)
}
