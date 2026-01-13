import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class GPGPUPropagationMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor
    (
        inputShape: [number, number, number], 
    ) 
    {
        const [inDepth, inHeight, inWidth] = inputShape
        this.outputShape = [inDepth, inHeight, inWidth]  
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float mmin(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        float mmax(float a, float b, float c, float d)
        {
            return max(max(max(a, b), c), d);
        }

        ivec3 getVoxelCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getVoxelValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        void main()
        {
            ivec3 voxelCoords = getVoxelCoords();

            float v0 = getVoxelValue(voxelCoords - ivec3(0,0,0));
            float v1 = getVoxelValue(voxelCoords - ivec3(1,0,0));
            float v2 = getVoxelValue(voxelCoords - ivec3(1,1,0));
            float v3 = getVoxelValue(voxelCoords - ivec3(1,0,1));
            float v4 = getVoxelValue(voxelCoords - ivec3(1,1,1));

            setOutput(max(v0, mmin(v1, v2, v3, v4)));
        }
        `
    }
}

class GPGPUOcclusionMap implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor
    (
        inputShape: [number, number, number], 
    ) 
    {
        const [inDepth, inHeight, inWidth] = inputShape
        const [outDepth, outHeight, outWidth] = [inDepth, inHeight, inWidth].map((x: number) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float mmin(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        float mmax(float a, float b, float c, float d)
        {
            return max(max(max(a, b), c), d);
        }

        ivec3 getCellCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getVoxelValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        float getPropagatedValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getB(safeCoords.z, safeCoords.y, safeCoords.x);
        }    

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            ivec3 voxelCoords = cellCoords - 1;
            
            float F000 = getVoxelValue(voxelCoords + ivec3(0,0,0));
            float F100 = getVoxelValue(voxelCoords + ivec3(1,0,0));
            float F010 = getVoxelValue(voxelCoords + ivec3(0,1,0));
            float F001 = getVoxelValue(voxelCoords + ivec3(0,0,1));
            float F011 = getVoxelValue(voxelCoords + ivec3(0,1,1));
            float F101 = getVoxelValue(voxelCoords + ivec3(1,0,1));
            float F110 = getVoxelValue(voxelCoords + ivec3(1,1,0));
            float F111 = getVoxelValue(voxelCoords + ivec3(1,1,1));

            float L000 = getPropagatedValue(voxelCoords + ivec3(0,0,0));
            float L010 = getPropagatedValue(voxelCoords + ivec3(0,1,0));
            float L001 = getPropagatedValue(voxelCoords + ivec3(0,0,1));
            float L011 = getPropagatedValue(voxelCoords + ivec3(0,1,1));

            float U000 = 0.0;
            float U010 = 0.0;
            float U001 = 0.0;
            float U011 = 0.0;

            U000 = max(U000, F100);
            U000 = max(U000, F110);
            U000 = max(U000, F101);
            U000 = max(U000, F111);
            U000 = max(U000, (F000 + F001 + F100) / 3.0);
            U000 = max(U000, (F000 + F010 + F100) / 3.0);
            U000 = max(U000, (F001 + F010 + F100) / 3.0);
            U000 = max(U000, (F001 + F100 + F101) / 3.0);
            U000 = max(U000, (F010 + F100 + F110) / 3.0);
            U000 = max(U000, (F011 + F101 + F110) / 3.0);

            U010 = max(U010, F110);
            U010 = max(U010, F111);
            U010 = max(U010, (F011 + F110 + F010) / 3.0);
            U010 = max(U010, (F011 + F110 + F111) / 3.0);
            
            U001 = max(U001, F101);
            U001 = max(U001, F111);
            U001 = max(U001, (F011 + F101 + F001) / 3.0);
            U001 = max(U001, (F011 + F101 + F111) / 3.0);

            U011 = max(U011, F111);

            bool occ = 
                L000 >= U000 && 
                L010 >= U010 && 
                L001 >= U001 && 
                L011 >= U011; 

            // bool occ = 
            //     mmin(L000, L010, L001, L011) >= 
            //     mmax(U000, U010, U001, U011);

            setOutput(float(occ));
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

export async function computeOcclusionMap(volumeMap: tf.Tensor3D) : Promise<tf.Tensor<tf.Rank>>
{
    const propagationProgram = new GPGPUPropagationMap(volumeMap.shape)
    const occlusionProgram = new GPGPUOcclusionMap(volumeMap.shape)

    let propagationMap = runProgram(propagationProgram, [volumeMap])
    let maxPropagation = Math.ceil(Math.max(...volumeMap.shape) / 2)
    console.log(maxPropagation)

    for (let i = 0; i < maxPropagation; i++) 
    {
        const prev = propagationMap
        propagationMap = runProgram(propagationProgram, [prev])
        prev.dispose()

        await tf.nextFrame()                     
    }

    const occlusionMap = runProgram(occlusionProgram, [volumeMap, propagationMap])
    propagationMap.dispose()

    return occlusionMap as tf.Tensor
}