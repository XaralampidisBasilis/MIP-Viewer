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


export class MaxPool3dProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

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
        const [filterDepth, filterHeight, filterWidth] = poolInfo.filterSize
        const [strideDepth, strideHeight, strideWidth] = poolInfo.strides
        const [padDepth, padHeight, padWidth] = poolInfo.pads

        this.outputShape = poolInfo.outputShape

        this.userCode = `
        const ivec3 inShape = ivec3(${inDepth}, ${inHeight}, ${inWidth});
        const ivec3 strides = ivec3(${strideDepth}, ${strideHeight}, ${strideWidth});
        const ivec3 pads = ivec3(${padDepth}, ${padHeight}, ${padWidth});

        void main()
        {
            ivec3 coords = getOutputCoords();

            int d0 = coords.x * strides.x - pads.x;
            int h0 = coords.y * strides.y - pads.y;
            int w0 = coords.z * strides.z - pads.z;

            float maxVal = -1.0 / 1e-20;
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

                        float value = getA(d, h, w);

                        if (found == 0.0 || value > maxVal)
                        {
                            maxVal = value;
                            found = 1.0;
                        }
                    }
                }
            }

            setOutput(maxVal);
        }
        `
    }
}


export class MinPool3dProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

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
        const [filterDepth, filterHeight, filterWidth] = poolInfo.filterSize
        const [strideDepth, strideHeight, strideWidth] = poolInfo.strides
        const [padDepth, padHeight, padWidth] = poolInfo.pads

        this.outputShape = poolInfo.outputShape

        this.userCode = `
        const ivec3 inShape = ivec3(${inDepth}, ${inHeight}, ${inWidth});
        const ivec3 strides = ivec3(${strideDepth}, ${strideHeight}, ${strideWidth});
        const ivec3 pads = ivec3(${padDepth}, ${padHeight}, ${padWidth});

        void main()
        {
            ivec3 coords = getOutputCoords();

            int d0 = coords.x * strides.x - pads.x;
            int h0 = coords.y * strides.y - pads.y;
            int w0 = coords.z * strides.z - pads.z;

            float minVal = 1.0 / 1e-20;
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

                        float value = getA(d, h, w);

                        if (found == 0.0 || value < minVal)
                        {
                            minVal = value;
                            found = 1.0;
                        }
                    }
                }
            }

            setOutput(minVal);
        }
        `
    }
}


export class AvgPool3dProgram implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

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
        const [filterDepth, filterHeight, filterWidth] = poolInfo.filterSize
        const [strideDepth, strideHeight, strideWidth] = poolInfo.strides
        const [padDepth, padHeight, padWidth] = poolInfo.pads

        this.outputShape = poolInfo.outputShape

        this.userCode = `
        const ivec3 inShape = ivec3(${inDepth}, ${inHeight}, ${inWidth});
        const ivec3 strides = ivec3(${strideDepth}, ${strideHeight}, ${strideWidth});
        const ivec3 pads = ivec3(${padDepth}, ${padHeight}, ${padWidth});

        void main()
        {
            ivec3 coords = getOutputCoords();

            int d0 = coords.x * strides.x - pads.x;
            int h0 = coords.y * strides.y - pads.y;
            int w0 = coords.z * strides.z - pads.z;

            float sumVal = 0.0;
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

                        float value = getA(d, h, w);
                        sumVal += value;
                        countVal += 1.0;
                    }
                }
            }

            setOutput(sumVal / countVal);
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
 * Max pooling for rank-3 tensors only: [depth, height, width]
 */
export function maxPool3d(
    inputTensor: tf.Tensor3D,
    filterSize: number | Pool3D,
    strides: number | Pool3D,
    pad: PadMode,
    dimRoundingMode?: DimRoundingMode
): tf.Tensor3D
{
    const program = new MaxPool3dProgram(
        inputTensor.shape as [number, number, number],
        filterSize,
        strides,
        pad,
        dimRoundingMode
    )

    return runProgram(program, [inputTensor], inputTensor.dtype) as tf.Tensor3D
}


/**
 * Min pooling for rank-3 tensors only: [depth, height, width]
 */
export function minPool3d(
    inputTensor: tf.Tensor3D,
    filterSize: number | Pool3D,
    strides: number | Pool3D,
    pad: PadMode,
    dimRoundingMode?: DimRoundingMode
): tf.Tensor3D
{
    const program = new MinPool3dProgram(
        inputTensor.shape as [number, number, number],
        filterSize,
        strides,
        pad,
        dimRoundingMode
    )

    return runProgram(program, [inputTensor], inputTensor.dtype) as tf.Tensor3D
}


/**
 * Average pooling for rank-3 tensors only: [depth, height, width]
 */
export function avgPool3d(
    inputTensor: tf.Tensor3D,
    filterSize: number | Pool3D,
    strides: number | Pool3D,
    pad: PadMode,
    dimRoundingMode?: DimRoundingMode
): tf.Tensor3D
{
    const program = new AvgPool3dProgram(
        inputTensor.shape as [number, number, number],
        filterSize,
        strides,
        pad,
        dimRoundingMode
    )

    return runProgram(program, [inputTensor], inputTensor.dtype) as tf.Tensor3D
}