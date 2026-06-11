import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

type Pool3D = [number, number, number]
type PadMode = 'valid' | 'same' | number
type DimRoundingMode = 'floor' | 'round' | 'ceil'


function normalize3Tuple(value: number | Pool3D): Pool3D
{
    if (typeof value === 'number')
    {
        return [value, value, value]
    }

    return value
}


function applyDimRoundingMode(value: number, mode?: DimRoundingMode): number
{
    if (mode == null) return Math.floor(value)

    switch (mode)
    {
        case 'floor': return Math.floor(value)
        case 'round': return Math.round(value)
        case 'ceil':  return Math.ceil(value)
    }
}


function computeOutSizeAndPad(
    inputSize: number,
    filterSize: number,
    stride: number,
    pad: PadMode,
    dimRoundingMode?: DimRoundingMode
): { outSize: number, padBefore: number }
{
    if (typeof pad === 'number')
    {
        const outSize = applyDimRoundingMode((inputSize + 2 * pad - filterSize) / stride + 1, dimRoundingMode)
        return { outSize: Math.max(0, outSize), padBefore: pad }
    }

    else if (pad === 'valid')
    {
        const outSize = Math.floor((inputSize - filterSize) / stride + 1)
        return { outSize: Math.max(0, outSize), padBefore: 0 }
    }
    
    else // if (pad === 'same')
    {
        const outSize = Math.ceil(inputSize / stride)
        const padNeeded = Math.max(0, (outSize - 1) * stride + filterSize - inputSize)
        const padBefore = Math.floor(padNeeded / 2)

        return { outSize, padBefore }
    }
}


function computePool3dInfo(
    inputShape: [number, number, number],
    filterSize: number | Pool3D,
    strides: number | Pool3D,
    pad: PadMode,
    dimRoundingMode?: DimRoundingMode
)
{
    const [inDepth, inHeight, inWidth] = inputShape
    const [filterDepth, filterHeight, filterWidth] = normalize3Tuple(filterSize)
    const [strideDepth, strideHeight, strideWidth] = normalize3Tuple(strides)

    if (typeof pad !== 'number' && dimRoundingMode != null)
    {
        throw new Error('dimRoundingMode requires numeric pad')
    }

    const depthInfo  = computeOutSizeAndPad(inDepth,  filterDepth,  strideDepth,  pad, dimRoundingMode)
    const heightInfo = computeOutSizeAndPad(inHeight, filterHeight, strideHeight, pad, dimRoundingMode)
    const widthInfo  = computeOutSizeAndPad(inWidth,  filterWidth,  strideWidth,  pad, dimRoundingMode)

    return {
        outputShape: [depthInfo.outSize, heightInfo.outSize, widthInfo.outSize] as [number, number, number],
        filterSize:  [filterDepth, filterHeight, filterWidth] as [number, number, number],
        strides:     [strideDepth, strideHeight, strideWidth] as [number, number, number],
        pads:        [depthInfo.padBefore, heightInfo.padBefore, widthInfo.padBefore] as [number, number, number]
    }
}


function validatePackedTensor5d(inputTensor: tf.Tensor5D)
{
    const shape = inputTensor.shape

    if (shape.length !== 5)
    {
        throw new Error(`inputTensor must be rank 5, got rank ${shape.length}`)
    }

    if (shape[3] !== 2 || shape[4] !== 2)
    {
        throw new Error(`inputTensor must have shape [D, H, W, 2, 2], got [${shape.join(', ')}]`)
    }
}


export class MaxPoolPacked3dProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        inputShape: [number, number, number],
        filterSize: number | Pool3D,
        strides: number | Pool3D,
        pad: PadMode,
        dimRoundingMode?: DimRoundingMode
    )
    {
        const [inDepth, inHeight, inWidth] = inputShape
        const poolInfo = computePool3dInfo(inputShape, filterSize, strides, pad, dimRoundingMode)
        const [outDepth, outHeight, outWidth] = poolInfo.outputShape
        const [filterDepth, filterHeight, filterWidth] = poolInfo.filterSize
        const [strideDepth, strideHeight, strideWidth] = poolInfo.strides
        const [padDepth, padHeight, padWidth] = poolInfo.pads

        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]

        this.userCode = `
        const ivec3 inShape = ivec3(${inDepth}, ${inHeight}, ${inWidth});
        const ivec3 strides = ivec3(${strideDepth}, ${strideHeight}, ${strideWidth});
        const ivec3 pads = ivec3(${padDepth}, ${padHeight}, ${padWidth});

        void main()
        {
            ivec5 coords = getOutputCoords();

            int d0 = coords.x * strides.x - pads.x;
            int h0 = coords.y * strides.y - pads.y;
            int w0 = coords.z * strides.z - pads.z;

            vec4 maxVal = vec4(-1.0 / 1e20);
            float found = 0.0;

            for (int kd = 0; kd < ${filterDepth}; ++kd)
            {
                int d = d0 + kd;
                if (d < 0 || d >= inShape.x) continue;

                for (int kh = 0; kh < ${filterHeight}; ++kh)
                {
                    int h = h0 + kh;
                    if (h < 0 || h >= inShape.y) continue;

                    for (int kw = 0; kw < ${filterWidth}; ++kw)
                    {
                        int w = w0 + kw;
                        if (w < 0 || w >= inShape.z) continue;

                        vec4 value = getA(d, h, w, 0, 0);

                        if (found == 0.0)
                        {
                            maxVal = value;
                            found = 1.0;
                        }
                        else
                        {
                            maxVal = max(maxVal, value);
                        }
                    }
                }
            }

            setOutput(maxVal);
        }
        `
    }
}


export class MinPoolPacked3dProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        inputShape: [number, number, number],
        filterSize: number | Pool3D,
        strides: number | Pool3D,
        pad: PadMode,
        dimRoundingMode?: DimRoundingMode
    )
    {
        const [inDepth, inHeight, inWidth] = inputShape
        const poolInfo = computePool3dInfo(inputShape, filterSize, strides, pad, dimRoundingMode)
        const [outDepth, outHeight, outWidth] = poolInfo.outputShape
        const [filterDepth, filterHeight, filterWidth] = poolInfo.filterSize
        const [strideDepth, strideHeight, strideWidth] = poolInfo.strides
        const [padDepth, padHeight, padWidth] = poolInfo.pads

        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]

        this.userCode = `
        const ivec3 inShape = ivec3(${inDepth}, ${inHeight}, ${inWidth});
        const ivec3 strides = ivec3(${strideDepth}, ${strideHeight}, ${strideWidth});
        const ivec3 pads = ivec3(${padDepth}, ${padHeight}, ${padWidth});

        void main()
        {
            ivec5 coords = getOutputCoords();

            int d0 = coords.x * strides.x - pads.x;
            int h0 = coords.y * strides.y - pads.y;
            int w0 = coords.z * strides.z - pads.z;

            vec4 minVal = vec4(1.0 / 1e20);
            float found = 0.0;

            for (int kd = 0; kd < ${filterDepth}; ++kd)
            {
                int d = d0 + kd;
                if (d < 0 || d >= inShape.x) continue;

                for (int kh = 0; kh < ${filterHeight}; ++kh)
                {
                    int h = h0 + kh;
                    if (h < 0 || h >= inShape.y) continue;

                    for (int kw = 0; kw < ${filterWidth}; ++kw)
                    {
                        int w = w0 + kw;
                        if (w < 0 || w >= inShape.z) continue;

                        vec4 value = getA(d, h, w, 0, 0);

                        if (found == 0.0)
                        {
                            minVal = value;
                            found = 1.0;
                        }
                        else
                        {
                            minVal = min(minVal, value);
                        }
                    }
                }
            }

            setOutput(minVal);
        }
        `
    }
}


export class AvgPoolPacked3dProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        inputShape: [number, number, number],
        filterSize: number | Pool3D,
        strides: number | Pool3D,
        pad: PadMode,
        dimRoundingMode?: DimRoundingMode
    )
    {
        const [inDepth, inHeight, inWidth] = inputShape
        const poolInfo = computePool3dInfo(inputShape, filterSize, strides, pad, dimRoundingMode)
        const [outDepth, outHeight, outWidth] = poolInfo.outputShape
        const [filterDepth, filterHeight, filterWidth] = poolInfo.filterSize
        const [strideDepth, strideHeight, strideWidth] = poolInfo.strides
        const [padDepth, padHeight, padWidth] = poolInfo.pads

        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]

        this.userCode = `
        const ivec3 inShape = ivec3(${inDepth}, ${inHeight}, ${inWidth});
        const ivec3 strides = ivec3(${strideDepth}, ${strideHeight}, ${strideWidth});
        const ivec3 pads = ivec3(${padDepth}, ${padHeight}, ${padWidth});

        void main()
        {
            ivec5 coords = getOutputCoords();

            int d0 = coords.x * strides.x - pads.x;
            int h0 = coords.y * strides.y - pads.y;
            int w0 = coords.z * strides.z - pads.z;

            vec4 sumVal = vec4(0.0);
            float countVal = 0.0;

            for (int kd = 0; kd < ${filterDepth}; ++kd)
            {
                int d = d0 + kd;
                if (d < 0 || d >= inShape.x) continue;

                for (int kh = 0; kh < ${filterHeight}; ++kh)
                {
                    int h = h0 + kh;
                    if (h < 0 || h >= inShape.y) continue;

                    for (int kw = 0; kw < ${filterWidth}; ++kw)
                    {
                        int w = w0 + kw;
                        if (w < 0 || w >= inShape.z) continue;

                        vec4 value = getA(d, h, w, 0, 0);
                        sumVal += value;
                        countVal += 1.0;
                    }
                }
            }

            setOutput(countVal > 0.0 ? sumVal / countVal : vec4(0.0));
        }
        `
    }
}


function runProgram(prog: GPGPUProgram, inputs: tf.Tensor[], dtype: tf.DataType = 'float32'): tf.Tensor
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.runWebGLProgram(prog, inputs, dtype)
    return tf.engine().makeTensorFromTensorInfo(info) as tf.Tensor
}


/**
 * Max pooling for packed rank-5 tensors of shape [depth, height, width, 2, 2]
 * Pooling is applied only over [depth, height, width].
 * Each logical voxel is a vec4 packed in the trailing [2, 2].
 */
export function maxPool3dPacked(
    inputTensor: tf.Tensor5D,
    filterSize: number | Pool3D,
    strides: number | Pool3D,
    pad: PadMode,
    dimRoundingMode?: DimRoundingMode
): tf.Tensor5D
{
    validatePackedTensor5d(inputTensor)

    const [inDepth, inHeight, inWidth] = inputTensor.shape as [number, number, number, number, number]

    const program = new MaxPoolPacked3dProgram(
        [inDepth, inHeight, inWidth],
        filterSize,
        strides,
        pad,
        dimRoundingMode
    )

    return runProgram(program, [inputTensor], inputTensor.dtype) as tf.Tensor5D
}


/**
 * Min pooling for packed rank-5 tensors of shape [depth, height, width, 2, 2]
 * Pooling is applied only over [depth, height, width].
 * Each logical voxel is a vec4 packed in the trailing [2, 2].
 */
export function minPool3dPacked(
    inputTensor: tf.Tensor5D,
    filterSize: number | Pool3D,
    strides: number | Pool3D,
    pad: PadMode,
    dimRoundingMode?: DimRoundingMode
): tf.Tensor5D
{
    validatePackedTensor5d(inputTensor)

    const [inDepth, inHeight, inWidth] = inputTensor.shape as [number, number, number, number, number]

    const program = new MinPoolPacked3dProgram(
        [inDepth, inHeight, inWidth],
        filterSize,
        strides,
        pad,
        dimRoundingMode
    )

    return runProgram(program, [inputTensor], inputTensor.dtype) as tf.Tensor5D
}


/**
 * Average pooling for packed rank-5 tensors of shape [depth, height, width, 2, 2]
 * Pooling is applied only over [depth, height, width].
 * Each logical voxel is a vec4 packed in the trailing [2, 2].
 */
export function avgPool3dPacked(
    inputTensor: tf.Tensor5D,
    filterSize: number | Pool3D,
    strides: number | Pool3D,
    pad: PadMode,
    dimRoundingMode?: DimRoundingMode
): tf.Tensor5D
{
    validatePackedTensor5d(inputTensor)

    const [inDepth, inHeight, inWidth] = inputTensor.shape as [number, number, number, number, number]

    const program = new AvgPoolPacked3dProgram(
        [inDepth, inHeight, inWidth],
        filterSize,
        strides,
        pad,
        dimRoundingMode
    )

    return runProgram(program, [inputTensor], inputTensor.dtype) as tf.Tensor5D
}