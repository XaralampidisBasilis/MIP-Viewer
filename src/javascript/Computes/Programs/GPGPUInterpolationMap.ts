import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class GPGPUInterpolationMap implements GPGPUProgram 
{
    variableNames = ['A'];
    outputShape: number[];
    userCode: string;
    packedInputs = false;
    packedOutput = false;

    constructor(inputShape: [number, number, number]) 
    {
        const [inDepth, inHeight, inWidth] = inputShape;
        this.outputShape = [inDepth, inHeight, inWidth, 4];
        this.userCode = `
        void main() 
        {
            ivec4 outputCoords = getOutputCoords();
            ivec3 voxelCoords = outputCoords.zyx;

            ivec2 leftRight = clamp(voxelCoords.x + ivec2(-1, 1), 0, ${inWidth  - 1});
            ivec2 topBottom = clamp(voxelCoords.y + ivec2(-1, 1), 0, ${inHeight - 1});
            ivec2 frontBack = clamp(voxelCoords.z + ivec2(-1, 1), 0, ${inDepth  - 1});

            float F = getA(voxelCoords.z, voxelCoords.y, voxelCoords.x);
            if (outputCoords.w == 3)
            {
                setOutput(F);
            }

            if (outputCoords.w == 2)
            {
                float Fzz = getA(frontBack.x, voxelCoords.y, voxelCoords.x) + 
                            getA(frontBack.y, voxelCoords.y, voxelCoords.x) - 
                            F * 2.0;
                            
                setOutput(Fzz);
            }

            if (outputCoords.w == 1)
            {
                float Fyy = getA(voxelCoords.z, topBottom.x, voxelCoords.x) + 
                            getA(voxelCoords.z, topBottom.y, voxelCoords.x) - 
                            F * 2.0;

                setOutput(Fyy);
            }

            if (outputCoords.w == 0)
            {
                float Fxx = getA(voxelCoords.z, voxelCoords.y, leftRight.x  ) + 
                            getA(voxelCoords.z, voxelCoords.y, leftRight.y  ) - 
                            F * 2.0;

                setOutput(Fxx);
            }
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

export function interpolationMap(inputTensor: tf.Tensor3D): tf.Tensor 
{
  const program = new GPGPUInterpolationMap(inputTensor.shape)
  return runProgram(program, [inputTensor]) as tf.Tensor4D
}
