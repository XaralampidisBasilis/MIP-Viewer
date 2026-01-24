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

        struct CellValues 
        { 
            float v000; 
            float v100; 
            float v010; 
            float v001; 
            float v011; 
            float v101; 
            float v110; 
            float v111; 
        }; 

        ivec3 getCellCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        float getVoxelValue(ivec3 voxelCoords)
        {
            ivec3 coords = clamp(voxelCoords, minVoxelCoords, maxVoxelCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getCellValues(in ivec3 cellCoords)
        {
            ivec3 coords = cellCoords - 1;
            
            CellValues c;
            c.v000 = getVoxelValue(coords + ivec3(0,0,0));
            c.v100 = getVoxelValue(coords + ivec3(1,0,0));
            c.v010 = getVoxelValue(coords + ivec3(0,1,0));
            c.v001 = getVoxelValue(coords + ivec3(0,0,1));
            c.v011 = getVoxelValue(coords + ivec3(0,1,1));
            c.v101 = getVoxelValue(coords + ivec3(1,0,1));
            c.v110 = getVoxelValue(coords + ivec3(1,1,0));
            c.v111 = getVoxelValue(coords + ivec3(1,1,1));
            return c;
        }

        float getMinOnRayMaxExitingFaceX(CellValues c000, CellValues c100)
        {
            float v0 = min4(c000.v100, c000.v110, c000.v101, c000.v111);
            float v1 = min4(c000.v100, c000.v110, c000.v101, c000.v011);
            float v2 = min4(c100.v001, c100.v010, c100.v100, c100.v011);

            v1 = min(v1, avg3(c000.v001, c000.v101, c000.v111));
            v1 = min(v1, avg3(c000.v001, c000.v011, c000.v111));
            v1 = min(v1, avg3(c000.v000, c000.v110, c000.v101));
            v1 = min(v1, avg3(c000.v010, c000.v001, c000.v111));
            v1 = min(v1, avg3(c000.v010, c000.v011, c000.v111));
            v1 = min(v1, avg3(c000.v010, c000.v110, c000.v111));

            v2 = min(v2, avg3(c100.v000, c100.v001, c100.v101));
            v2 = min(v2, avg3(c100.v000, c100.v100, c100.v101));
            v2 = min(v2, avg3(c100.v000, c100.v010, c100.v110));
            v2 = min(v2, avg3(c100.v000, c100.v100, c100.v110));
            v2 = min(v2, avg3(c100.v010, c100.v001, c100.v111));
            v2 = min(v2, avg3(c100.v000, c100.v110, c100.v101));

            return max3(v0, v1, v2);
        }
    
        float getMinOnRayMaxExitingFaceY(CellValues c000, CellValues c010)
        {
            float v0 = min4(c000.v010, c000.v011, c000.v110, c000.v111);
            float v1 = min4(c000.v010, c000.v011, c000.v110, c000.v101);
            float v2 = min4(c010.v100, c010.v001, c010.v010, c010.v101);

            v1 = min(v1, avg3(c000.v100, c000.v110, c000.v111));
            v1 = min(v1, avg3(c000.v100, c000.v101, c000.v111));
            v1 = min(v1, avg3(c000.v000, c000.v011, c000.v110));
            v1 = min(v1, avg3(c000.v001, c000.v100, c000.v111));
            v1 = min(v1, avg3(c000.v001, c000.v101, c000.v111));
            v1 = min(v1, avg3(c000.v001, c000.v011, c000.v111));

            v2 = min(v2, avg3(c010.v000, c010.v100, c010.v110));
            v2 = min(v2, avg3(c010.v000, c010.v010, c010.v110));
            v2 = min(v2, avg3(c010.v000, c010.v001, c010.v011));
            v2 = min(v2, avg3(c010.v000, c010.v010, c010.v011));
            v2 = min(v2, avg3(c010.v001, c010.v100, c010.v111));
            v2 = min(v2, avg3(c010.v000, c010.v011, c010.v110));

            return max3(v0, v1, v2);
        }

        float getMinOnRayMaxExitingFaceZ(CellValues c000, CellValues c001)
        {
            float v0 = min4(c000.v001, c000.v011, c000.v101, c000.v111);
            float v1 = min4(c000.v001, c000.v011, c000.v101, c000.v110);
            float v2 = min4(c001.v100, c001.v010, c001.v001, c001.v110);

            v1 = min(v1, avg3(c000.v100, c000.v101, c000.v111));
            v1 = min(v1, avg3(c000.v100, c000.v110, c000.v111));
            v1 = min(v1, avg3(c000.v000, c000.v011, c000.v101));
            v1 = min(v1, avg3(c000.v010, c000.v100, c000.v111));
            v1 = min(v1, avg3(c000.v010, c000.v110, c000.v111));
            v1 = min(v1, avg3(c000.v010, c000.v011, c000.v111));

            v2 = min(v2, avg3(c001.v000, c001.v100, c001.v101));
            v2 = min(v2, avg3(c001.v000, c001.v001, c001.v101));
            v2 = min(v2, avg3(c001.v000, c001.v010, c001.v011));
            v2 = min(v2, avg3(c001.v000, c001.v001, c001.v011));
            v2 = min(v2, avg3(c001.v010, c001.v100, c001.v111));
            v2 = min(v2, avg3(c001.v000, c001.v011, c001.v101));

            return max3(v0, v1, v2);
        }

        void main()
        {
            ivec3 cellCoords = getCellCoords();

            CellValues c000 = getCellValues(cellCoords + ivec3(0,0,0));
            CellValues c100 = getCellValues(cellCoords + ivec3(1,0,0));
            CellValues c010 = getCellValues(cellCoords + ivec3(0,1,0));
            CellValues c001 = getCellValues(cellCoords + ivec3(0,0,1));

            float xMin = getMinOnRayMaxExitingFaceX(c000, c100);
            float yMin = getMinOnRayMaxExitingFaceY(c000, c010);
            float zMin = getMinOnRayMaxExitingFaceZ(c000, c001);

            setOutput(vec4(xMin, yMin, zMin, 0.0));
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
        this.outputShape = [outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec2 minCellCoords = ivec2(0);
        const ivec2 maxCellCoords = ivec2(${outWidth-1}, ${outHeight-1});

        float min3(float a, float b, float c) { return min(min(a, b), c); }

        ivec2 getCellCoords()
        {
            ivec4 outputCoords = getOutputCoords();
            return ivec2(outputCoords.y, outputCoords.x);
        }

        vec4 getCellValues(ivec2 cellCoords)
        {
            ivec2 coords = clamp(cellCoords, minCellCoords, maxCellCoords);
            return getA(coords.y, coords.x, 0, 0);
        }

        vec4 getPrevCellValues(ivec2 cellCoords)
        {
            ivec2 coords = clamp(cellCoords, minCellCoords, maxCellCoords);
            return getB(coords.y, coords.x, 0, 0);
        }
                
        void main()
        {
            ivec2 coords = getCellCoords();

            vec4 c111 = getCellValues(coords - ivec2(0,0));
            vec4 c011 = getCellValues(coords - ivec2(1,0));
            vec4 c101 = getCellValues(coords - ivec2(0,1));
            vec4 c001 = getCellValues(coords - ivec2(1,1));

            vec4 c110 = getPrevCellValues(coords - ivec2(0,0));
            vec4 c010 = getPrevCellValues(coords - ivec2(1,0));
            vec4 c100 = getPrevCellValues(coords - ivec2(0,1));
            vec4 c000 = getPrevCellValues(coords - ivec2(1,1));

            float c001x = max(c001.x, c000.z);
            float c001y = max(c001.y, c000.z);
            float c011x = max(c011.x, min(c001y, c010.z));
            float c101y = max(c101.y, min(c001x, c100.z));

            c111.x = max(c111.x, min(c110.z, max(c101.y, c100.z)));
            c111.y = max(c111.y, min(c110.z, max(c011.x, c010.z)));
            c111.z = max(c111.z, min3(c011x, c101y, c110.z));

            setOutput(c111);
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
    const startPropagationMap = new GPGPUStartPropagationMap(volumeMap.shape)
    const updatePropagationSlice = new GPGPUUpdatePropagationSlice(volumeMap.shape)

    const propagationMap = runProgram(startPropagationMap, [volumeMap])
    console.log(propagationMap.mean().dataSync())
    const propagationSlices = tf.unstack(propagationMap, 0)
    propagationMap.dispose()

    for (let i = 1; i < propagationSlices.length; i++)
    {
        const updatedSlice = runProgram(updatePropagationSlice, [propagationSlices[i], propagationSlices[i-1]])
        propagationSlices[i].dispose()
        propagationSlices[i] = updatedSlice
    }

    const propagatedMap = tf.stack(propagationSlices, 0)
    console.log(propagatedMap.mean().dataSync())
    tf.dispose(propagationSlices)

    return propagatedMap as tf.Tensor
}