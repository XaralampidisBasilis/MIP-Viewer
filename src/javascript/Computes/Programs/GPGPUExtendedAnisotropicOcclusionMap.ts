import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class GPGPUStartOcclusionMap implements GPGPUProgram 
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
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 voxelMinCoords = ivec3(0);
        const ivec3 voxelMaxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        struct CellSamples
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

        vec3 packVec3ToHalf2x16(vec3 a, vec3 b)
        {
            uvec3 p;
            p.x = packHalf2x16(vec2(a.x, b.x));
            p.y = packHalf2x16(vec2(a.y, b.y));
            p.z = packHalf2x16(vec2(a.z, b.z));

            return uintBitsToFloat(p);
        }

        ivec3 getCellCoords()
        {
            ivec5 outputCoords = getOutputCoords();
            return ivec3(outputCoords.z, outputCoords.y, outputCoords.x);
        }

        float getVoxelSample(ivec3 voxelCoords)
        {
            ivec3 voxelCoordsSafe = clamp(voxelCoords, voxelMinCoords, voxelMaxCoords);
            return getA(voxelCoordsSafe.z, voxelCoordsSafe.y, voxelCoordsSafe.x);
        }

        CellSamples getCellSamples(in ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;
            
            CellSamples s;
            s.f000 = getVoxelSample(voxelCoords + ivec3(0,0,0));
            s.f100 = getVoxelSample(voxelCoords + ivec3(1,0,0));
            s.f010 = getVoxelSample(voxelCoords + ivec3(0,1,0));
            s.f001 = getVoxelSample(voxelCoords + ivec3(0,0,1));
            s.f011 = getVoxelSample(voxelCoords + ivec3(0,1,1));
            s.f101 = getVoxelSample(voxelCoords + ivec3(1,0,1));
            s.f110 = getVoxelSample(voxelCoords + ivec3(1,1,0));
            s.f111 = getVoxelSample(voxelCoords + ivec3(1,1,1));
            return s;
        }

        vec3 getCellMinOutputs(CellSamples s)
        {
            float xMinOutput = min(min(min(s.f100, s.f110), s.f101), s.f111);
            float yMinOutput = min(min(min(s.f010, s.f110), s.f011), s.f111);
            float zMinOutput = min(min(min(s.f001, s.f101), s.f011), s.f111);
            return vec3(xMinOutput, yMinOutput, zMinOutput);
        }

        vec3 getCellMaxOutputs(CellSamples s)
        {
            float xMaxOutput = max(max(max(s.f100, s.f110), s.f101), s.f111);
            float yMaxOutput = max(max(max(s.f010, s.f110), s.f011), s.f111);
            float zMaxOutput = max(max(max(s.f001, s.f101), s.f011), s.f111);
            return vec3(xMaxOutput, yMaxOutput, zMaxOutput);
        }

        void main()
        {
            ivec3 cellCoords = getCellCoords();

            CellSamples cellSamples = getCellSamples(cellCoords);

            vec3 cellMinOutputs = getCellMinOutputs(cellSamples);
            vec3 cellMaxOutputs = getCellMaxOutputs(cellSamples);

            vec3 cellMinMaxOutputs = packVec3ToHalf2x16(cellMinOutputs, cellMaxOutputs);

            setOutput(vec4(cellMinMaxOutputs, 0.0));
        }
        `
    }
}

class GPGPUUpdateOcclusionMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor
    (
        inputShape: [number, number, number], 
    ) 
    {
        const [inDepth, inHeight, inWidth] = inputShape
        const [outDepth, outHeight, outWidth] = [inDepth, inHeight, inWidth].map((x: number) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 cellMinCoords = ivec3(0);
        const ivec3 cellMaxCoords = ivec3(${outDepth-1}, ${outDepth-1}, ${outDepth-1});

        float unpackHalfFloatLow(float a)
        {
            return unpackHalf2x16(floatBitsToUint(a)).x;
        }

        float unpackHalfFloatHigh(float a)
        {
            return unpackHalf2x16(floatBitsToUint(a)).y;
        }

        vec3 packVec3ToHalf2x16(vec3 a, vec3 b)
        {
            uvec3 p;
            p.x = packHalf2x16(vec2(a.x, b.x));
            p.y = packHalf2x16(vec2(a.y, b.y));
            p.z = packHalf2x16(vec2(a.z, b.z));

            return uintBitsToFloat(p);
        }

        ivec3 getCellCoords()
        {
            ivec5 outputCoords = getOutputCoords();
            return ivec3(outputCoords.z, outputCoords.y, outputCoords.x);
        }

        vec4 getCellData(ivec3 cellCoords)
        {
            ivec3 cellCoordsSafe = clamp(cellCoords, cellMinCoords, cellMaxCoords);
            return getA(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0);
        }

        vec3 getCellMinInputs(ivec3 cellCoords)
        {
            float xMinInput = unpackHalfFloatLow(getCellData(cellCoords - ivec3(1,0,0)).x);
            float yMinInput = unpackHalfFloatLow(getCellData(cellCoords - ivec3(0,1,0)).y);
            float zMinInput = unpackHalfFloatLow(getCellData(cellCoords - ivec3(0,0,1)).z);
            return vec3(xMinInput, yMinInput, zMinInput);
        }

        vec3 getCellMinOutputs(vec3 cellMinMaxOutputs)
        {
            float xMinOutput = unpackHalfFloatLow(cellMinMaxOutputs.x);
            float yMinOutput = unpackHalfFloatLow(cellMinMaxOutputs.y);
            float zMinOutput = unpackHalfFloatLow(cellMinMaxOutputs.z);
            return vec3(xMinOutput, yMinOutput, zMinOutput);
        }

        vec3 getCellMaxOutputs(vec3 cellMinMaxOutputs)
        {
            float xMaxOutput = unpackHalfFloatHigh(cellMinMaxOutputs.x);
            float yMaxOutput = unpackHalfFloatHigh(cellMinMaxOutputs.y);
            float zMaxOutput = unpackHalfFloatHigh(cellMinMaxOutputs.z);
            return vec3(xMaxOutput, yMaxOutput, zMaxOutput);
        }

        void updateCellMinOutputs(vec3 cellMinInputs, inout vec3 cellMinOutputs)
        {
            cellMinOutputs.x = max(cellMinOutputs.x, min(min(cellMinInputs.x, cellMinInputs.y), cellMinInputs.z));
            cellMinOutputs.y = max(cellMinOutputs.x, min(cellMinInputs.x, cellMinInputs.z));
            cellMinOutputs.z = max(cellMinOutputs.x, min(cellMinInputs.x, cellMinInputs.y));
        }

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            vec4 cellData = getCellData(cellCoords);

            vec3 cellMinMaxOutputs = cellData.xyz;
            bool cellOccluded = bool(cellData.w);

            vec3 cellMinInputs = getCellMinInputs(cellCoords);
            vec3 cellMinOutputs = getCellMinOutputs(cellMinMaxOutputs);
            vec3 cellMaxOutputs = getCellMaxOutputs(cellMinMaxOutputs);

            cellOccluded = cellOccluded || all(greaterThanEqual(cellMinInputs, cellMaxOutputs));
            
            updateCellMinOutputs(cellMinInputs, cellMinOutputs);
            cellMinMaxOutputs = packVec3ToHalf2x16(cellMinOutputs, cellMaxOutputs);

            setOutput(vec4(cellMinMaxOutputs, cellOccluded));
        }
        `
    }
}

class GPGPUEndOcclusionMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
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
        ivec3 getCellCoords()
        {
            ivec3 outputCoords = getOutputCoords();
            return ivec3(outputCoords.z, outputCoords.y, outputCoords.x);
        }

        float getCellOcclusion(ivec3 cellCoords)
        {
            return getA(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0).w;
        }

        void main()
        {
            ivec3 cellCoords = getCellCoords();

            setOutput(getCellOcclusion(cellCoords));
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

export async function computeExtendedAnisotropicOcclusionMap(interpolationMap: tf.Tensor3D) : Promise<tf.Tensor<tf.Rank>>
{
    const startProgram  = new GPGPUStartOcclusionMap(interpolationMap.shape)
    const updateProgram = new GPGPUUpdateOcclusionMap(interpolationMap.shape)
    const endProgram    = new GPGPUEndOcclusionMap(interpolationMap.shape)

    let updateMap = runProgram(startProgram, [interpolationMap])

    for (let i = 0; i < 100; i++) 
    {
        const prev = updateMap
        updateMap = runProgram(updateProgram, [prev])
        prev.dispose()

        await tf.nextFrame()                     
    }

    const endMap = runProgram(endProgram, [updateMap])
    tf.dispose(updateMap)

    return endMap as tf.Tensor
}