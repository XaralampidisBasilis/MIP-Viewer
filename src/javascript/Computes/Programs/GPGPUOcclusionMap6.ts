import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class GPGPUStartPropagationMap implements GPGPUProgram 
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
        const [outDepth, outHeight, outWidth] = inputShape.map((x: number) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 minVoxelCoords = ivec3(0);
        const ivec3 maxVoxelCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float avg3(float a, float b, float c) { return (a + b + c) * (1.0 / 3.0); }
        float max3(float a, float b, float c) { return max(max(a, b), c); }
        float min4(float a, float b, float c, float d) { return min(min(min(a, b), c), d); }

        float getVoxelValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minVoxelCoords, maxVoxelCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        ivec3 getCellCoords()
        {
            ivec5 outputCoords = getOutputCoords();
            return ivec3(outputCoords.z, outputCoords.y, outputCoords.x);
        }

        struct CellValues 
        { 
            float f000; 
            float f100; 
            float f010; 
            float f001; 
            float f011; 
            float f101; 
            float f110; 
            float f111; 
        };

        CellValues getCurrentSliceCellOutputs(in ivec3 cellCoords)
        {
            ivec3 coords = cellCoords - 1;
            
            CellValues C;
            C.f000 = getVoxelValue(coords + ivec3(0,0,0));
            C.f100 = getVoxelValue(coords + ivec3(1,0,0));
            C.f010 = getVoxelValue(coords + ivec3(0,1,0));
            C.f001 = getVoxelValue(coords + ivec3(0,0,1));
            C.f011 = getVoxelValue(coords + ivec3(0,1,1));
            C.f101 = getVoxelValue(coords + ivec3(1,0,1));
            C.f110 = getVoxelValue(coords + ivec3(1,1,0));
            C.f111 = getVoxelValue(coords + ivec3(1,1,1));
            return C;
        }

        float getMinOnRayMaxExitingFaceX(CellValues C000, CellValues C100)
        {
            float minOnFace       = min4(C000.f100, C000.f110, C000.f101, C000.f111);
            float minOnBeforeFace = min4(C000.f100, C000.f110, C000.f101, C000.f011);
            float minOnAfterFace  = min4(C100.f001, C100.f010, C100.f100, C100.f011);

            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f001, C000.f101, C000.f111));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f001, C000.f011, C000.f111));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f000, C000.f110, C000.f101));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f010, C000.f001, C000.f111));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f010, C000.f011, C000.f111));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f010, C000.f110, C000.f111));

            minOnAfterFace = min(minOnAfterFace, avg3(C100.f000, C100.f001, C100.f101));
            minOnAfterFace = min(minOnAfterFace, avg3(C100.f000, C100.f100, C100.f101));
            minOnAfterFace = min(minOnAfterFace, avg3(C100.f000, C100.f010, C100.f110));
            minOnAfterFace = min(minOnAfterFace, avg3(C100.f000, C100.f100, C100.f110));
            minOnAfterFace = min(minOnAfterFace, avg3(C100.f010, C100.f001, C100.f111));
            minOnAfterFace = min(minOnAfterFace, avg3(C100.f000, C100.f110, C100.f101));

            return max3(minOnAfterFace, minOnFace, minOnBeforeFace);
        }
    
        float getMinOnRayMaxExitingFaceY(CellValues C000, CellValues C010)
        {
            float minOnFace       = min4(C000.f010, C000.f011, C000.f110, C000.f111);
            float minOnBeforeFace = min4(C000.f010, C000.f011, C000.f110, C000.f101);
            float minOnAfterFace  = min4(C010.f100, C010.f001, C010.f010, C010.f101);

            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f100, C000.f110, C000.f111));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f100, C000.f101, C000.f111));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f000, C000.f011, C000.f110));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f001, C000.f100, C000.f111));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f001, C000.f101, C000.f111));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f001, C000.f011, C000.f111));

            minOnAfterFace = min(minOnAfterFace, avg3(C010.f000, C010.f100, C010.f110));
            minOnAfterFace = min(minOnAfterFace, avg3(C010.f000, C010.f010, C010.f110));
            minOnAfterFace = min(minOnAfterFace, avg3(C010.f000, C010.f001, C010.f011));
            minOnAfterFace = min(minOnAfterFace, avg3(C010.f000, C010.f010, C010.f011));
            minOnAfterFace = min(minOnAfterFace, avg3(C010.f001, C010.f100, C010.f111));
            minOnAfterFace = min(minOnAfterFace, avg3(C010.f000, C010.f011, C010.f110));

            return max3(minOnAfterFace, minOnFace, minOnBeforeFace);
        }

        float getMinOnRayMaxExitingFaceZ(CellValues C000, CellValues C001)
        {
            float minOnFace       = min4(C000.f001, C000.f011, C000.f101, C000.f111);
            float minOnBeforeFace = min4(C000.f001, C000.f011, C000.f101, C000.f110);
            float minOnAfterFace  = min4(C001.f100, C001.f010, C001.f001, C001.f110);

            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f100, C000.f101, C000.f111));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f100, C000.f110, C000.f111));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f000, C000.f011, C000.f101));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f010, C000.f100, C000.f111));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f010, C000.f110, C000.f111));
            minOnBeforeFace = min(minOnBeforeFace, avg3(C000.f010, C000.f011, C000.f111));

            minOnAfterFace = min(minOnAfterFace, avg3(C001.f000, C001.f100, C001.f101));
            minOnAfterFace = min(minOnAfterFace, avg3(C001.f000, C001.f001, C001.f101));
            minOnAfterFace = min(minOnAfterFace, avg3(C001.f000, C001.f010, C001.f011));
            minOnAfterFace = min(minOnAfterFace, avg3(C001.f000, C001.f001, C001.f011));
            minOnAfterFace = min(minOnAfterFace, avg3(C001.f010, C001.f100, C001.f111));
            minOnAfterFace = min(minOnAfterFace, avg3(C001.f000, C001.f011, C001.f101));

            return max3(minOnAfterFace, minOnFace, minOnBeforeFace);
        }

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            CellValues C000 = getCurrentSliceCellOutputs(cellCoords + ivec3(0,0,0));
            CellValues C100 = getCurrentSliceCellOutputs(cellCoords + ivec3(1,0,0));
            CellValues C010 = getCurrentSliceCellOutputs(cellCoords + ivec3(0,1,0));
            CellValues C001 = getCurrentSliceCellOutputs(cellCoords + ivec3(0,0,1));

            float xMinMax = getMinOnRayMaxExitingFaceX(C000, C100);
            float yMinMax = getMinOnRayMaxExitingFaceY(C000, C010);
            float zMinMax = getMinOnRayMaxExitingFaceZ(C000, C001);

            setOutput(vec4(xMinMax, yMinMax, zMinMax, 0.0));
        }
        `
    }
}

class GPGPUUpdatePropagationSlice implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor
    (
        inputShape: [number, number, number], 
    ) 
    {
        const [outDepth, outHeight, outWidth] = inputShape.map((x: number) => x + 1)
        this.outputShape = [1, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 minCellCoords = ivec3(0);
        const ivec3 maxCellCoords = ivec3(${outDepth-1}, ${outDepth-1}, ${outDepth-1});

        float min3(float a, float b, float c) { return min(min(a, b), c); }

        ivec3 getCellCoords()
        {
            ivec5 outputCoords = getOutputCoords();
            return ivec3(outputCoords.z, outputCoords.y, outputCoords.x);
        }

        vec4 getA(ivec3 cellCoords)
        {
            ivec3 safeCoords = clamp(cellCoords, minCellCoords, maxCellCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x, 0, 0);
        }

        vec4 getB(ivec3 cellCoords)
        {
            ivec3 safeCoords = clamp(cellCoords, minCellCoords, maxCellCoords);
            return getB(safeCoords.z, safeCoords.y, safeCoords.x, 0, 0);
        }
                
        vec3 getMinOnRayMaxEnteringCell(ivec3 cellCoords) 
        { 
            float xMin = getA(cellCoords - ivec3(1,0,0)).x; 
            float yMin = getA(cellCoords - ivec3(0,1,0)).y; 
            float zMin = getB(cellCoords).z; 

            return vec3(xMin, yMin, zMin);
        }

        vec3 getMinOnRayMaxExitingCell(ivec3 cellCoords) 
        { 
            return getA(cellCoords).xyz; 
        }

        void main()
        {
            ivec3 cellCoords = getCellCoords();

            vec3 minOutputs = getMinOnRayMaxExitingCell(cellCoords);
            vec3 minInputs = getMinOnRayMaxEnteringCell(cellCoords);

            minOutputs.x = max(minOutputs.x, min(minInputs.y, minInputs.z));
            minOutputs.y = max(minOutputs.y, min(minInputs.x, minInputs.z));
            minOutputs.z = max(minOutputs.z, min3(minInputs.x, minInputs.y, minInputs.z));

            setOutput(vec4(minOutputs, 0.0));
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

export async function computeExtendedAnisotropicOcclusionMap(volumeMap: tf.Tensor3D) : Promise<tf.Tensor<tf.Rank>>
{
    const startPropagationMap = new GPGPUStartPropagationMap(volumeMap.shape)
    const updatePropagationSlice = new GPGPUUpdatePropagationSlice(volumeMap.shape)

    const propagationMap = runProgram(startPropagationMap, [volumeMap])
    const propagationSlices = tf.split(propagationMap, propagationMap.shape[0], 0)

    let previousSlice = propagationSlices[0]

    for (let i = 0; i < propagationSlices.length; i++)
    {
        let propagationSlice = propagationSlices[i]
        propagationSlice = runProgram(updatePropagationSlice, [propagationSlice, previousSlice])
        propagationSlice = runProgram(updatePropagationSlice, [propagationSlice, previousSlice])
        
        previousSlice = propagationSlice
        propagationSlices[i] = propagationSlice
       
    }

    return propagationMap as tf.Tensor
}