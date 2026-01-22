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
        float min4(float a, float b, float c, float d) { return min(min(min(a, b), c), d); }

        vec3 packVec3ToHalf2x16(vec3 u, vec3 v)
        {
            uvec3 p;
            p.x = packHalf2x16(vec2(u.x, v.x));
            p.y = packHalf2x16(vec2(u.y, v.y));
            p.z = packHalf2x16(vec2(u.z, v.z));

            return uintBitsToFloat(p);
        }

        ivec3 getCellCoords()
        {
            ivec5 outputCoords = getOutputCoords();
            return ivec3(outputCoords.z, outputCoords.y, outputCoords.x);
        }

        float getVoxelValue(ivec3 voxelCoords)
        {
            ivec3 voxelCoordsSafe = clamp(voxelCoords, minVoxelCoords, maxVoxelCoords);
            return getA(voxelCoordsSafe.z, voxelCoordsSafe.y, voxelCoordsSafe.x);
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

        CellValues getCellValues(in ivec3 cellCoords)
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

        // Lower bound for:  min_over_rays  max_over_t f(o + t d)
        // Rays enter the cell through the face x=0 and directions are dy/dx, dz/dx in [0, 1]
        // Any such ray must intersect:
        //  - the entry face x=0 -> bilinear min
        //  - the interior surface x = min(1-y, 1-z) -> bounded via triangle Bernstein minima
        // Since the ray maximum is >= the value at each intersection, a guaranteed bound is:
        // max(min_on_face,  min_on_surface).
     
        float getMinOnRayMaxEnteringFaceX0(CellValues C)
        {
            float minOnFace = min4(C.f100, C.f110, C.f101, C.f111);
            float minOnSurf = min4(C.f010, C.f001, C.f011, C.f100);
    
            minOnSurf = min(minOnSurf, avg3(C.f000, C.f010, C.f110));
            minOnSurf = min(minOnSurf, avg3(C.f000, C.f100, C.f110));
            minOnSurf = min(minOnSurf, avg3(C.f000, C.f001, C.f101));
            minOnSurf = min(minOnSurf, avg3(C.f000, C.f100, C.f101));
            minOnSurf = min(minOnSurf, avg3(C.f001, C.f010, C.f111));
            minOnSurf = min(minOnSurf, avg3(C.f000, C.f101, C.f110));

            return max(minOnFace, minOnSurf);
        }

        float getMinOnRayMaxEnteringFaceY0(CellValues C)
        {   
            CellValues newC = CellValues(C.f000, C.f010, C.f100, C.f110, C.f001, C.f011, C.f101, C.f111); // transform (x,y,z) -> (y,x,z)
            return getMinOnRayMaxEnteringFaceX0(newC);
        }

        float getMinOnRayMaxEnteringFaceZ0(in CellValues C)
        {
            CellValues newC = CellValues(C.f000, C.f001, C.f010, C.f011, C.f100, C.f101, C.f110, C.f111); // transform (x,y,z) -> (z,y,x)
            return getMinOnRayMaxEnteringFaceX0(newC);
        }

        float getMinOnRayMaxExitingFaceX1(CellValues C)
        {   
            CellValues newC = CellValues(C.f111, C.f011, C.f101, C.f001, C.f110, C.f010, C.f100, C.f000); // transform (x,y,z) -> (1-x,1-y,1-z)
            return getMinOnRayMaxEnteringFaceX0(newC);
        }

        float getMinOnRayMaxExitingFaceY1(CellValues C)
        {   
            CellValues newC = CellValues(C.f111, C.f101, C.f011, C.f001, C.f110, C.f100, C.f010, C.f000); // transform (x,y,z) -> (1-y,1-x,1-z)
            return getMinOnRayMaxEnteringFaceX0(newC);
        }

        float getMinOnRayMaxExitingFaceZ1(CellValues C)
        {   
            CellValues newC = CellValues(C.f111, C.f110, C.f101, C.f100, C.f011, C.f010, C.f001, C.f000); // transform (x,y,z) -> (1-z,1-y,1-x)
            return getMinOnRayMaxEnteringFaceX0(newC);
        }

        vec3 getMinOnRayMaxEnteringFaces(CellValues C)
        {
            float xMinMax = getMinOnRayMaxEnteringFaceX0(C);
            float yMinMax = getMinOnRayMaxEnteringFaceY0(C);
            float zMinMax = getMinOnRayMaxEnteringFaceZ0(C);

            return vec3(xMinMax, yMinMax, zMinMax);
        }

        vec3 getMinOnRayMaxExitingFaces(CellValues C)
        {
            float xMinMax = getMinOnRayMaxExitingFaceX1(C);
            float yMinMax = getMinOnRayMaxExitingFaceY1(C);
            float zMinMax = getMinOnRayMaxExitingFaceZ1(C);

            return vec3(xMinMax, yMinMax, zMinMax);
        }

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            CellValues cellValues = getCellValues(cellCoords);

            vec3 minOnRayMaxEnteringFaces = getMinOnRayMaxEnteringFaces(cellValues);
            vec3 minOnRayMaxExitingFaces = getMinOnRayMaxExitingFaces(cellValues);

            vec3 minOnRayMaxEnteringExitingFaces = packVec3ToHalf2x16(minOnRayMaxEnteringFaces, minOnRayMaxExitingFaces);
            setOutput(vec4(minOnRayMaxEnteringExitingFaces, 0.0));
        }
        `
    }
}

class GPGPUUpdatePropagationSlice implements GPGPUProgram 
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