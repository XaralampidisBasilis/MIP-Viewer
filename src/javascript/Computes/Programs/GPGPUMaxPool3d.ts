import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

export class GPGPUMaxPool3d implements GPGPUProgram 
{
    variableNames = ['A']
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

        int startIdx(int outIndex, int inSize, int outSize) 
        {
            return int(floor(float(outIndex) * float(inSize) / float(outSize)));
        }

        int endIdx(int outIndex, int inSize, int outSize) 
        {
            return int(ceil(float(outIndex + 1) * float(inSize) / float(outSize)));
        }

        bool singular(int d0, int d1, int h0, int h1, int w0, int w1)
        {
            return d1 <= d0 || h1 <= h0 || w1 <= w0; 
        }

        void main() 
        {
            ivec3 coords = getOutputCoords();

            int d0 = startIdx(coords.x, inShape.x, outShape.x);
            int d1 = endIdx  (coords.x, inShape.x, outShape.x);
            int h0 = startIdx(coords.y, inShape.y, outShape.y);
            int h1 = endIdx  (coords.y, inShape.y, outShape.y);
            int w0 = startIdx(coords.z, inShape.z, outShape.z);
            int w1 = endIdx  (coords.z, inShape.z, outShape.z);

            d0 = clamp(d0, 0, inShape.x);
            d1 = clamp(d1, 0, inShape.x);
            h0 = clamp(h0, 0, inShape.y);
            h1 = clamp(h1, 0, inShape.y);
            w0 = clamp(w0, 0, inShape.z);
            w1 = clamp(w1, 0, inShape.z);

            if (singular(d0, d1, h0, h1, w0, w1))
            {
                int d = clamp(d0, 0, inShape.x-1);
                int h = clamp(h0, 0, inShape.y-1);
                int w = clamp(w0, 0, inShape.z-1);

                setOutput(getA(d, h, w));
                return;
            }

            // Unrolled loops with runtime guards to only cover [start,end)
            float maxVal = -1.0 / 0.0;

            for (int k = 0; k < ${maxK}; ++k) 
            {
                int d = d0 + k;
                if (d >= d1) break;

                for (int j = 0; j < ${maxJ}; ++j) 
                {
                    int h = h0 + j;
                    if (h >= h1) break;

                    for (int i = 0; i < ${maxI}; ++i) 
                    {
                        int w = w0 + i;
                        if (w >= w1) break;

                        float value = getA(d, h, w);
                        maxVal = max(maxVal, value);
                    }
                }
            }

            setOutput(maxVal);
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
export function maxPool3d(inputTensor: tf.Tensor3D, outputShape: [number, number, number]): tf.Tensor3D 
{
  if (sameShape(inputTensor.shape, outputShape)) return inputTensor
  const program = new GPGPUMaxPool3d(inputTensor.shape as [number, number, number], outputShape)
  return runProgram(program, [inputTensor]) as tf.Tensor3D
}
