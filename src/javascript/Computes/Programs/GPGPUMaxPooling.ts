import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

export class GPGPUMaxPooling implements GPGPUProgram 
{
    variableNames = ['Input']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        inputShape: [number, number, number], 
        outputShape: [number, number, number]
    ) 
    {
        const [inDepth, inHeight, inWidth] = inputShape
        const [outDepth, outHeight, outWidth] = outputShape
        this.outputShape = outputShape

        const maxK = Math.ceil(inDepth  / outDepth ) + 1
        const maxJ = Math.ceil(inHeight / outHeight) + 1
        const maxI = Math.ceil(inWidth  / outWidth ) + 1

        this.userCode = `
        const ivec3 inShape  = ivec3(${inDepth},  ${inHeight},  ${inWidth});
        const ivec3 outShape = ivec3(${outDepth}, ${outHeight}, ${outWidth});

        int startIndex(int outIndex, int inSize, int outSize) 
        {
            return int(floor(float(outIndex) * float(inSize) / float(outSize)));
        }

        int endIndex(int outIndex, int inSize, int outSize) 
        {
            return int(ceil(float(outIndex + 1) * float(inSize) / float(outSize)));
        }

        bool singular(int depthStart, int depthEnd, int heightStart, int heightEnd, int widthStart, int widthEnd)
        {
            return depthEnd <= depthStart || heightEnd <= heightStart || widthEnd <= widthStart; 
        }

        void main() 
        {
            ivec3 outCoords = getOutputCoords();

            int depthStart  = startIndex(outCoords.x, inShape.x, outShape.x);
            int depthEnd    = endIndex  (outCoords.x, inShape.x, outShape.x);
            int heightStart = startIndex(outCoords.y, inShape.y, outShape.y);
            int heightEnd   = endIndex  (outCoords.y, inShape.y, outShape.y);
            int widthStart  = startIndex(outCoords.z, inShape.z, outShape.z);
            int widthEnd    = endIndex  (outCoords.z, inShape.z, outShape.z);

            depthStart  = clamp(depthStart , 0, inShape.x);
            depthEnd    = clamp(depthEnd   , 0, inShape.x);
            heightStart = clamp(heightStart, 0, inShape.y);
            heightEnd   = clamp(heightEnd  , 0, inShape.y);
            widthStart  = clamp(widthStart , 0, inShape.z);
            widthEnd    = clamp(widthEnd   , 0, inShape.z);

            if (singular(depthStart, depthEnd, heightStart, heightEnd, widthStart, widthEnd))
            {
                int depth  = clamp(depthStart , 0, inShape.x-1);
                int height = clamp(heightStart, 0, inShape.y-1);
                int width  = clamp(widthStart , 0, inShape.z-1);

                setOutput(getInput(depth, height, width));
                return;
            }

            // Unrolled loops with runtime guards to only cover [start,end)
            float maxValue = -1.0/0.0;

            for (int k = 0; k < ${maxK}; ++k) 
            {
                int depth = depthStart + k;
                if (depth >= depthEnd) break;

                for (int j = 0; j < ${maxJ}; ++j) 
                {
                    int height = heightStart + j;
                    if (height >= heightEnd) break;

                    for (int i = 0; i < ${maxI}; ++i) 
                    {
                        int width = widthStart + i;
                        if (width >= widthEnd) break;

                        float value = getInput(depth, height, width);
                        maxValue = max(maxValue, value);
                    }
                }
            }

            setOutput(maxValue);
        }
        `
    }
}


function runProgram(prog: GPGPUProgram, inputs: tf.Tensor[]): tf.Tensor
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(prog, inputs)
    return tf.engine().makeTensorFromTensorInfo(info) as tf.Tensor
}

function sameShape(a: number[], b: number[]) 
{
    return a.length === b.length && a.every((v, i) => v === b[i])
}

/**
 * True (adaptive) max pooling from inputShape -> outputShape.
 * Each output voxel is max over its mapped input box.
 */
export function maxPooling3d(inputTensor: tf.Tensor3D, outputShape: [number, number, number]): tf.Tensor3D 
{
  if (sameShape(inputTensor.shape, outputShape)) return inputTensor
  const program = new GPGPUMaxPooling(inputTensor.shape as [number, number, number], outputShape)
  return runProgram(program, [inputTensor]) as tf.Tensor3D
}
