import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class GPGPUOcclusionMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor
    (
        inputShape: [number, number, number], 
    ) 
    {
        const [inDepth, inHeight, inWidth] = inputShape
        const [outDepth, outHeight, outWidth] = [inDepth, inHeight, inWidth].map((x: number) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth]     
        this.userCode = `

        const ivec3 voxelMinCoords = ivec3(0);
        const ivec3 voxelMaxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float getVoxelSample(ivec3 voxelCoords)
        {
            voxelCoords = clamp(voxelCoords, voxelMinCoords, voxelMaxCoords);
            return getA(voxelCoords.z, voxelCoords.y, voxelCoords.x);
        }

        bool getCellOcclusion(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            float f000 = getVoxelSample(voxelCoords + ivec3(0, 0, 0));
            float f100 = getVoxelSample(voxelCoords + ivec3(1, 0, 0));
            float f010 = getVoxelSample(voxelCoords + ivec3(0, 1, 0));
            float f001 = getVoxelSample(voxelCoords + ivec3(0, 0, 1));
            float f011 = getVoxelSample(voxelCoords + ivec3(0, 1, 1));
            float f101 = getVoxelSample(voxelCoords + ivec3(1, 0, 1));
            float f110 = getVoxelSample(voxelCoords + ivec3(1, 1, 0));
            float f111 = getVoxelSample(voxelCoords + ivec3(1, 1, 1));

            if (f100 - f000 <= 0.0) return false;
            if (f101 - f001 <= 0.0) return false;
            if (f110 - f010 <= 0.0) return false;
            if (f111 - f011 <= 0.0) return false;

            if (f101*2.0 - f100 - f001 <= 0.0) return false;
            if (f110*2.0 - f100 - f010 <= 0.0) return false;
            if (f111*2.0 - f101 - f011 <= 0.0) return false;
            if (f111*2.0 - f110 - f011 <= 0.0) return false;
            if (f001 + f100 - f000*2.0 <= 0.0) return false;
            if (f010 + f100 - f000*2.0 <= 0.0) return false;
            if (f011 + f101 - f001*2.0 <= 0.0) return false;
            if (f011 + f110 - f010*2.0 <= 0.0) return false;

            if (f011 + f101 - f001 - f000 <= 0.0) return false;
            if (f011 + f110 - f010 - f000 <= 0.0) return false;
            if (f011 + f111 - f010 - f001 <= 0.0) return false;
            if (f101 + f110 - f100 - f000 <= 0.0) return false;
            if (f101 + f111 - f100 - f001 <= 0.0) return false;
            if (f110 + f111 - f100 - f010 <= 0.0) return false;
            if (f001 + f010 + f100 - f000*3.0 <= 0.0) return false;
            if (f111*3.0 - f101 - f110 - f011 <= 0.0) return false;
            
            return true;
        } 

        bool getCellOcclusion2(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            float f000 = getVoxelSample(voxelCoords + ivec3(0, 0, 0));
            float f100 = getVoxelSample(voxelCoords + ivec3(1, 0, 0));
            float f010 = getVoxelSample(voxelCoords + ivec3(0, 1, 0));
            float f001 = getVoxelSample(voxelCoords + ivec3(0, 0, 1));
            float f011 = getVoxelSample(voxelCoords + ivec3(0, 1, 1));
            float f101 = getVoxelSample(voxelCoords + ivec3(1, 0, 1));
            float f110 = getVoxelSample(voxelCoords + ivec3(1, 1, 0));
            float f111 = getVoxelSample(voxelCoords + ivec3(1, 1, 1));
            
            return true;
        } 
        
        void main()
        {
            ivec3 outputCoords = getOutputCoords();
            ivec3 cellCoords = outputCoords.zyx;
            
            bool cellOcclusion = getCellOcclusion3(cellCoords);

            setOutput(vec4(cellOcclusion));
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

export function computeOcclusionMap(interpolationMap: tf.Tensor3D) : tf.Tensor
{
    const program = new GPGPUOcclusionMap(interpolationMap.shape)
    return runProgram(program, [interpolationMap]) as tf.Tensor3D
}