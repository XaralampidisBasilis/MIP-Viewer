import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { unstackPacked } from './unstack_packed_keepDims_webgl'
import { stackPacked } from './stack_packed_keepDims_webgl'

type Axis = 0 | 1 | 2
type Permute = [Axis, Axis, Axis]
type Reverse = Axis[]

function inversePermute(permute: Permute): Permute
{
    const inv = new Array<number>(permute.length)
    for (let i = 0; i < permute.length; i++) 
    {
        inv[permute[i]] = i
    }

    return inv as Permute
}

function complementReverse(reverse: Reverse): Reverse 
{
    const set = new Set<Axis>(reverse)
    const complement: Reverse = []

    for (const axis of [0, 1, 2] as const) 
    {
        if (!set.has(axis)) complement.push(axis)
    }
    return complement
}

class GPGPUUnidirectionalMinimaMapDeprecated implements GPGPUProgram 
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
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

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

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        ivec3 getVoxelCoords(ivec3 coords, int ox, int oy, int oz)
        {
            return coords + ivec3(ox, oy, oz);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec3 coords)
        {
            CellValues c;

            coords = coords - 1;

            c.v000 = getA(getVoxelCoords(coords, 0,0,0));
            c.v100 = getA(getVoxelCoords(coords, 1,0,0));
            c.v010 = getA(getVoxelCoords(coords, 0,1,0));
            c.v001 = getA(getVoxelCoords(coords, 0,0,1));
            c.v011 = getA(getVoxelCoords(coords, 0,1,1));
            c.v101 = getA(getVoxelCoords(coords, 1,0,1));
            c.v110 = getA(getVoxelCoords(coords, 1,1,0));
            c.v111 = getA(getVoxelCoords(coords, 1,1,1));

            return c;
        }

        float getMinOnFaceX(CellValues c000)
        {           
            return min4(c000.v100, c000.v110, c000.v101, c000.v111);
        }

        float getMinOnFaceY(CellValues c000)
        {
            return min4(c000.v010, c000.v110, c000.v011, c000.v111);
        }
            
        float getMinOnFaceZ(CellValues c000)
        {
            return min4(c000.v001, c000.v011, c000.v101, c000.v111);
        }

        void main()
        {
            ivec3 coords = getOutCoords();
            CellValues c = getValues(coords);

            float xMin = getMinOnFaceX(c);
            float yMin = getMinOnFaceY(c);
            float zMin = getMinOnFaceZ(c);

            setOutput(vec4(xMin, yMin, zMin, 0.0));
        }
        `
    }
}

class GPGPUUnidirectionalMaximaMapDeprecated implements GPGPUProgram 
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
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float avg3(float a, float b, float c) { return (a + b + c) * (1.0 / 3.0); }
        float max4(float a, float b, float c, float d) { return max(max(max(a, b), c), d); }

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

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        ivec3 getVoxelCoords(ivec3 coords, int ox, int oy, int oz)
        {
            return coords + ivec3(ox, oy, oz);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec3 coords)
        {
            CellValues c;

            coords = coords - 1;

            c.v000 = getA(getVoxelCoords(coords, 0,0,0));
            c.v100 = getA(getVoxelCoords(coords, 1,0,0));
            c.v010 = getA(getVoxelCoords(coords, 0,1,0));
            c.v001 = getA(getVoxelCoords(coords, 0,0,1));
            c.v011 = getA(getVoxelCoords(coords, 0,1,1));
            c.v101 = getA(getVoxelCoords(coords, 1,0,1));
            c.v110 = getA(getVoxelCoords(coords, 1,1,0));
            c.v111 = getA(getVoxelCoords(coords, 1,1,1));

            return c;
        }

        float getMaxOnFaceX(CellValues c)
        {
            return max4(c.v100, c.v110, c.v101, c.v111);
        }
    
        float getMaxOnFaceY(CellValues c)
        {
            return max4(c.v010, c.v110, c.v011, c.v111);
        }

        float getMaxOnFaceZ(CellValues c)
        {
            return max4(c.v001, c.v011, c.v101, c.v111);
        }

        void main()
        {
            ivec3 cellCoords = getOutCoords();
            CellValues c = getValues(cellCoords);

            float xMax = getMaxOnFaceX(c);
            float yMax = getMaxOnFaceY(c);
            float zMax = getMaxOnFaceZ(c);

            setOutput(vec4(xMax, yMax, zMax, 1.0));
        }
        `
    }
}

class GPGPUUnidirectionalMinimaMap implements GPGPUProgram 
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
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float avg3(float a, float b, float c) { return (a + b + c) * (1.0/3.0); }
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

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        ivec3 getVoxelCoords(ivec3 coords, int ox, int oy, int oz)
        {
            return coords + ivec3(ox, oy, oz);
        }

        ivec3 getCellCoords(ivec3 coords, int ox, int oy, int oz)
        {
            return coords + ivec3(ox, oy, oz);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec3 coords)
        {
            CellValues c;

            coords = coords - 1;

            c.v000 = getA(getVoxelCoords(coords, 0,0,0));
            c.v100 = getA(getVoxelCoords(coords, 1,0,0));
            c.v010 = getA(getVoxelCoords(coords, 0,1,0));
            c.v001 = getA(getVoxelCoords(coords, 0,0,1));
            c.v011 = getA(getVoxelCoords(coords, 0,1,1));
            c.v101 = getA(getVoxelCoords(coords, 1,0,1));
            c.v110 = getA(getVoxelCoords(coords, 1,1,0));
            c.v111 = getA(getVoxelCoords(coords, 1,1,1));

            return c;
        }

        float getMinOnFaceX(CellValues c000, CellValues c100)
        {
            float m0, m1, m2;

            m0 = min4(c000.v100, c000.v110, c000.v101, c000.v111);
            m1 = min4(c000.v100, c000.v110, c000.v101, c000.v011);
            m2 = min4(c100.v001, c100.v010, c100.v100, c100.v011);

            m1 = min(m1, avg3(c000.v001, c000.v101, c000.v111));
            m1 = min(m1, avg3(c000.v001, c000.v011, c000.v111));
            m1 = min(m1, avg3(c000.v000, c000.v101, c000.v110));
            m1 = min(m1, avg3(c000.v001, c000.v010, c000.v111));
            m1 = min(m1, avg3(c000.v010, c000.v110, c000.v111));
            m1 = min(m1, avg3(c000.v010, c000.v011, c000.v111));

            m2 = min(m2, avg3(c100.v000, c100.v010, c100.v110));
            m2 = min(m2, avg3(c100.v000, c100.v100, c100.v110));
            m2 = min(m2, avg3(c100.v000, c100.v001, c100.v101));
            m2 = min(m2, avg3(c100.v000, c100.v100, c100.v101));
            m2 = min(m2, avg3(c100.v001, c100.v010, c100.v111));
            m2 = min(m2, avg3(c100.v000, c100.v101, c100.v110));

            return max3(m0, m1, m2);
        }

        float getMinOnFaceY(CellValues c000, CellValues c010)
        {
            float m0, m1, m2;

            m0 = min4(c000.v010, c000.v110, c000.v011, c000.v111);
            m1 = min4(c000.v010, c000.v110, c000.v011, c000.v101);  
            m2 = min4(c010.v001, c010.v100, c010.v010, c010.v101);

            m1 = min(m1, avg3(c000.v001, c000.v011, c000.v111));
            m1 = min(m1, avg3(c000.v001, c000.v101, c000.v111));
            m1 = min(m1, avg3(c000.v000, c000.v011, c000.v110));
            m1 = min(m1, avg3(c000.v001, c000.v100, c000.v111));
            m1 = min(m1, avg3(c000.v100, c000.v110, c000.v111));
            m1 = min(m1, avg3(c000.v100, c000.v101, c000.v111));

            m2 = min(m2, avg3(c010.v000, c010.v100, c010.v110));
            m2 = min(m2, avg3(c010.v000, c010.v010, c010.v110));
            m2 = min(m2, avg3(c010.v000, c010.v001, c010.v011));
            m2 = min(m2, avg3(c010.v000, c010.v010, c010.v011));
            m2 = min(m2, avg3(c010.v001, c010.v100, c010.v111));
            m2 = min(m2, avg3(c010.v000, c010.v011, c010.v110));

            return max3(m0, m1, m2);
        }
            
        float getMinOnFaceZ(CellValues c000, CellValues c001)
        {
            float m0, m1, m2;

            m0 = min4(c000.v001, c000.v011, c000.v101, c000.v111);
            m1 = min4(c000.v001, c000.v011, c000.v101, c000.v110);
            m2 = min4(c001.v100, c001.v010, c001.v001, c001.v110);

            m1 = min(m1, avg3(c000.v100, c000.v101, c000.v111));
            m1 = min(m1, avg3(c000.v100, c000.v110, c000.v111));
            m1 = min(m1, avg3(c000.v000, c000.v011, c000.v101));
            m1 = min(m1, avg3(c000.v010, c000.v100, c000.v111));
            m1 = min(m1, avg3(c000.v010, c000.v110, c000.v111));
            m1 = min(m1, avg3(c000.v010, c000.v011, c000.v111));

            m2 = min(m2, avg3(c001.v000, c001.v100, c001.v101));
            m2 = min(m2, avg3(c001.v000, c001.v001, c001.v101));
            m2 = min(m2, avg3(c001.v000, c001.v010, c001.v011));
            m2 = min(m2, avg3(c001.v000, c001.v001, c001.v011));
            m2 = min(m2, avg3(c001.v010, c001.v100, c001.v111));
            m2 = min(m2, avg3(c001.v000, c001.v011, c001.v101));

            return max3(m0, m1, m2);
        }

        void main()
        {
            ivec3 outCoords = getOutCoords();

            CellValues c000 = getValues(getCellCoords(outCoords, 0,0,0));
            CellValues c100 = getValues(getCellCoords(outCoords, 1,0,0));
            CellValues c010 = getValues(getCellCoords(outCoords, 0,1,0));
            CellValues c001 = getValues(getCellCoords(outCoords, 0,0,1));

            float xMin = getMinOnFaceX(c000, c100);
            float yMin = getMinOnFaceY(c000, c010);
            float zMin = getMinOnFaceZ(c000, c001);

            setOutput(vec4(xMin, yMin, zMin, 0.0));
        }
        `
    }
}

class GPGPUUnidirectionalMaximaMap implements GPGPUProgram 
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
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float avg3(float a, float b, float c) { return (a + b + c) * (1.0 / 3.0); }

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

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        ivec3 getVoxelCoords(ivec3 coords, int ox, int oy, int oz)
        {
            return  coords + ivec3(ox, oy, oz);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec3 coords)
        {
            CellValues c;

            coords = coords - 1;

            c.v000 = getA(getVoxelCoords(coords, 0,0,0));
            c.v100 = getA(getVoxelCoords(coords, 1,0,0));
            c.v010 = getA(getVoxelCoords(coords, 0,1,0));
            c.v001 = getA(getVoxelCoords(coords, 0,0,1));
            c.v011 = getA(getVoxelCoords(coords, 0,1,1));
            c.v101 = getA(getVoxelCoords(coords, 1,0,1));
            c.v110 = getA(getVoxelCoords(coords, 1,1,0));
            c.v111 = getA(getVoxelCoords(coords, 1,1,1));

            return c;
        }

        float getMaxOnFaceX(CellValues c)
        {
            float m = -1.0;

            m = max(m, avg3(c.v000, c.v001, c.v100));
            m = max(m, avg3(c.v001, c.v010, c.v100));
            m = max(m, avg3(c.v010, c.v011, c.v110));
            m = max(m, avg3(c.v001, c.v100, c.v101));
            m = max(m, avg3(c.v011, c.v101, c.v110));
            m = max(m, avg3(c.v011, c.v110, c.v111));
            m = max(m, c.v000);
            m = max(m, c.v001);
            m = max(m, c.v010);
            m = max(m, c.v011);
            m = max(m, c.v101);
            m = max(m, c.v111);
            
            return m;
        }
    
        float getMaxOnFaceY(CellValues c)
        {
            float m = -1.0;

            m = max(m, avg3(c.v000, c.v001, c.v010));
            m = max(m, avg3(c.v001, c.v010, c.v011));
            m = max(m, avg3(c.v001, c.v010, c.v100));
            m = max(m, avg3(c.v011, c.v101, c.v110));
            m = max(m, avg3(c.v100, c.v101, c.v110));
            m = max(m, avg3(c.v101, c.v110, c.v111));
            m = max(m, c.v000);
            m = max(m, c.v001);
            m = max(m, c.v011);
            m = max(m, c.v100);
            m = max(m, c.v101);
            m = max(m, c.v111);
        
            return m;
        }

        float getMaxOnFaceZ(CellValues c)
        {
            float m = -1.0;

            m = max(m, c.v000);
            m = max(m, c.v100);
            m = max(m, c.v010);
            m = max(m, c.v001);
            m = max(m, c.v011);
            m = max(m, c.v101);
            m = max(m, c.v110);
            m = max(m, c.v111);

            return m;
        }

        float getMaxOnCell(CellValues c)
        {
            float m = -1.0;

            m = max(m, avg3(c.v001, c.v010, c.v011) - c.v000);
            m = max(m, avg3(c.v001, c.v010, c.v100) - c.v000);
            m = max(m, avg3(c.v001, c.v100, c.v101) - c.v000);
            m = max(m, avg3(c.v011, c.v101, c.v110) - c.v000);
            m = max(m, avg3(c.v011, c.v110, c.v111) - c.v010);
            m = max(m, avg3(c.v101, c.v110, c.v111) - c.v100);
            m = max(m, avg3(c.v001, c.v010, c.v000) - c.v000);
            m = max(m, avg3(c.v001, c.v100, c.v000) - c.v000);
            m = max(m, avg3(c.v011, c.v110, c.v010) - c.v010);
            m = max(m, avg3(c.v101, c.v110, c.v100) - c.v100);
            m = max(m, c.v001 - c.v000);
            m = max(m, c.v011 - c.v000);
            m = max(m, c.v011 - c.v010);
            m = max(m, c.v101 - c.v000);
            m = max(m, c.v111 - c.v000);
            m = max(m, c.v111 - c.v010);
            m = max(m, c.v101 - c.v100);
            m = max(m, c.v111 - c.v100);
            m = max(m, c.v111 - c.v110);

            return m;
        }

        void main()
        {
            ivec3 cellCoords = getOutCoords();
            CellValues c = getValues(cellCoords);

            float xMax = getMaxOnFaceX(c);
            float yMax = getMaxOnFaceY(c);
            float zMax = getMaxOnFaceZ(c);
            float wMax = getMaxOnCell(c);

            setOutput(vec4(xMax, yMax, zMax, wMax));
        }
        `
    }
}

class GPGPUUpdateUnidirectionalMinimaSlices implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(outputShape: [number, number, number, 2, 2]) 
    {
        const [outDepth, outHeight, outWidth] = outputShape
        this.outputShape = outputShape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        float min3(float a, float b, float c) { return min(min(a, b), c); }

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        ivec3 getCellCoords(ivec3 coords, int ox, int oy, int oz)
        {
            return coords + ivec3(ox, oy, oz);
        }

        vec4 getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x, 0, 0);
        }

        vec4 getB(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getB(coords.z, coords.y, coords.x, 0, 0);
        }

        float getMinOnFaceX(vec4 c111, vec4 c110, vec4 c101, vec4 c100)
        {
            float t10 = c100.z;
            
            t10 = max(c101.y, t10);
            t10 = min(c110.z, t10);
            t10 = max(c111.x, t10);

            return t10;
        }

        float getMinOnFaceY(vec4 c111, vec4 c110, vec4 c011, vec4 c010)
        {
            float t01 = c010.z;

            t01 = max(c011.x, t01);
            t01 = min(c110.z, t01);
            t01 = max(c111.y, t01);

            return t01;
        }

        float getMinOnFaceZ(vec4 c111, vec4 c110, vec4 c101, vec4 c011, vec4 c100, vec4 c010, vec4 c001, vec4 c000)
        {
            float t00, t01, t10, t11;
            t00 = c000.z;

            t01 = max(c001.y, t00);
            t01 = min(c010.z, t01);
            t01 = max(c011.x, t01);

            t10 = max(c001.x, t00);
            t10 = min(c100.z, t10);
            t10 = max(c101.y, t10);

            t11 = min(t01, t10);
            t11 = min(c110.z, t11);
            t11 = max(c111.z, t11);

            return t11;
        }
                
        void main()
        {
            ivec3 cellCoords = getOutCoords();

            vec4 c111 = getA(getCellCoords(cellCoords, -0,-0,-0));
            vec4 c011 = getA(getCellCoords(cellCoords, -1,-0,-0));
            vec4 c101 = getA(getCellCoords(cellCoords, -0,-1,-0));
            vec4 c001 = getA(getCellCoords(cellCoords, -1,-1,-0));
            vec4 c110 = getB(getCellCoords(cellCoords, -0,-0,-1));
            vec4 c010 = getB(getCellCoords(cellCoords, -1,-0,-1));
            vec4 c100 = getB(getCellCoords(cellCoords, -0,-1,-1));
            vec4 c000 = getB(getCellCoords(cellCoords, -1,-1,-1));

            c111.x = getMinOnFaceX(c111, c110, c101, c100);
            c111.y = getMinOnFaceY(c111, c110, c011, c010);
            c111.z = getMinOnFaceZ(c111, c110, c101, c011, c100, c010, c001, c000);

            setOutput(c111);
        }
        `
    }
}

class GPGPUUpdateUnidirectionalMinimaMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(outputShape: [number, number, number, 2, 2]) 
    {
        const [outDepth, outHeight, outWidth] = outputShape.slice(0,3)
        this.outputShape = outputShape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        float min3(float a, float b, float c) { return min(min(a, b), c); }

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        ivec3 getCellCoords(ivec3 coords, int ox, int oy, int oz)
        {
            return coords + ivec3(ox, oy, oz);
        }

        vec4 getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x, 0, 0);
        }


        float getMinOnFaceX(vec4 c111, vec4 c110, vec4 c101, vec4 c100)
        {
            float t10 = c100.z;
            
            t10 = max(c101.y, t10);
            t10 = min(c110.z, t10);
            t10 = max(c111.x, t10);

            return t10;
        }

        float getMinOnFaceY(vec4 c111, vec4 c110, vec4 c011, vec4 c010)
        {
            float t01 = c010.z;

            t01 = max(c011.x, t01);
            t01 = min(c110.z, t01);
            t01 = max(c111.y, t01);

            return t01;
        }

        float getMinOnFaceZ(vec4 c111, vec4 c110, vec4 c101, vec4 c011, vec4 c100, vec4 c010, vec4 c001, vec4 c000)
        {
            float t00, t01, t10, t11;
            t00 = c000.z;

            t01 = max(c001.y, t00);
            t01 = min(c010.z, t01);
            t01 = max(c011.x, t01);

            t10 = max(c001.x, t00);
            t10 = min(c100.z, t10);
            t10 = max(c101.y, t10);

            t11 = min(t01, t10);
            t11 = min(c110.z, t11);
            t11 = max(c111.z, t11);

            return t11;
        }
                
        void main()
        {
            ivec3 coords = getOutCoords();

            vec4 c111 = getA(getCellCoords(coords, -0,-0,-0));
            vec4 c011 = getA(getCellCoords(coords, -1,-0,-0));
            vec4 c101 = getA(getCellCoords(coords, -0,-1,-0));
            vec4 c001 = getA(getCellCoords(coords, -1,-1,-0));
            vec4 c110 = getA(getCellCoords(coords, -0,-0,-1));
            vec4 c010 = getA(getCellCoords(coords, -1,-0,-1));
            vec4 c100 = getA(getCellCoords(coords, -0,-1,-1));
            vec4 c000 = getA(getCellCoords(coords, -1,-1,-1));

            c111.x = getMinOnFaceX(c111, c110, c101, c100);
            c111.y = getMinOnFaceY(c111, c110, c011, c010);
            c111.z = getMinOnFaceZ(c111, c110, c101, c011, c100, c010, c001, c000);

            setOutput(c111);
        }
        `
    }
}

class GPGPUUnidirectionalOcclusionMap implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = false

    constructor(outputShape: [number, number, number]) 
    {
        const [outDepth, outHeight, outWidth] = outputShape
        this.outputShape = outputShape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        ivec3 getOutCoords()
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x, 0, 0);
        }

        vec4 getB(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getB(coords.z, coords.y, coords.x, 0, 0);
        }

        bool getOcclusion(ivec3 coords)
        {
            vec4 minValues = getA(coords);
            vec4 maxValues = getB(coords);

            bvec4 tests = greaterThanEqual(minValues, maxValues);
            bool occlusion = all(tests.xyz) || tests.w;

            return occlusion;
        }
                
        void main()
        {
            ivec3 coords = getOutCoords();

            setOutput(float(getOcclusion(coords)));
        }
        `
    }
}

class GPGPUBidirectionalOcclusionMap implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(outputShape: [number, number, number], ) 
    {
        const [outDepth, outHeight, outWidth] = outputShape
        this.outputShape = outputShape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        uvec4 toUint(vec4 v) { return uvec4(round(v)) & 1u; }

        ivec3 getOutCoords()
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        vec4 getB(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getB(coords.z, coords.y, coords.x);
        }
                
        void main()
        {
            ivec3 coords = getOutCoords();

            uvec4 oA = toUint(getA(coords));
            uvec4 oB = toUint(getB(coords));

            setOutput(vec4(oA | oB));
        }
        `
    }
}

class GPGPUAnisotropicBidirectionalOcclusionMap implements GPGPUProgram 
{
    variableNames = ['A', 'B', 'C', 'D']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(outputShape: [number, number, number], ) 
    {
        const [outDepth, outHeight, outWidth] = outputShape
        this.outputShape = outputShape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        uvec4 toUint(vec4 v) { return uvec4(round(v)) & 1u; }

        ivec3 getOutCoords()
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        vec4 getB(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getB(coords.z, coords.y, coords.x);
        }

        vec4 getC(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getC(coords.z, coords.y, coords.x);
        }

        vec4 getD(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getD(coords.z, coords.y, coords.x);
        }

        uvec4 bitpack(uvec4 oA, uvec4 oB, uvec4 oC, uvec4 oD) 
        { 
            return (oA << 0) | (oB << 1) | (oC << 2) | (oD << 3);
        }
 
        void main()
        {
            ivec3 coords = getOutCoords();

            uvec4 oA = toUint(getA(coords));
            uvec4 oB = toUint(getB(coords));
            uvec4 oC = toUint(getC(coords));
            uvec4 oD = toUint(getD(coords));

            setOutput(vec4(bitpack(oA, oB, oC, oD)));
        }
        `
    }
}

class GPGPUExtendedAnisotropicBidirectionalOcclusionMap implements GPGPUProgram 
{
    variableNames = ['A', 'B', 'C']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(outputShape: [number, number, number]) 
    {
        const [outDepth, outHeight, outWidth] = outputShape
        this.outputShape = outputShape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});
           
        uvec4 toUint(vec4 v) { return uvec4(round(v)) & 15u; }

        ivec3 getOutCoords()
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        vec4 getB(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getB(coords.z, coords.y, coords.x);
        }

        vec4 getC(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getC(coords.z, coords.y, coords.x);
        }

        ivec4 bitpack(uvec4 oA, uvec4 oB, uvec4 oC)
        {
            uvec4 p = (oA << 0u) | (oB << 4u) | (oC << 8u); // 0..4095
            return ivec4(p) - ivec4(2048); // -2048..2047 in half float precision 
        }
 
        void main()
        {
            ivec3 coords = getOutCoords();

            uvec4 oA = toUint(getA(coords));
            uvec4 oB = toUint(getB(coords));
            uvec4 oC = toUint(getC(coords));

            setOutput(vec4(bitpack(oA, oB, oC)));
        }
        `
    }
}

class GPGPUUnpackFromAnisotropicBidirectionalOcclusionMap implements GPGPUProgram 
{
    variableNames = ['A'] 
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'map', type: 'int' as const }]

    constructor(outputShape: [number, number, number]) 
    {
        const [outDepth, outHeight, outWidth] = outputShape
        this.outputShape = outputShape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        uvec4 toUint(vec4 v) { return uvec4(round(v)) & 15u; }

        ivec3 getOutCoords() 
        {
            ivec3 c = getOutputCoords();
            return ivec3(c.z, c.y, c.x);
        }

        vec4 getA(ivec3 coords) 
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        void main() 
        {
            ivec3 coords = getOutCoords();

            uvec4 u = toUint(getA(coords));
            uvec4 o = (u >> map) & 1u;

            setOutput(vec4(o));
        }
        `
    }
}

class GPGPUUnpackFromExtendedAnisotropicBidirectionalOcclusionMap implements GPGPUProgram 
{
    variableNames = ['A'] 
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'map', type: 'int' as const }]

    constructor(outputShape: [number, number, number]) 
    {
        const [outDepth, outHeight, outWidth] = outputShape
        this.outputShape = outputShape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        ivec4 toInt(vec4 v) { return clamp(ivec4(round(v)), -2048, 2047); }

        ivec3 getOutCoords() 
        {
            ivec3 c = getOutputCoords();
            return ivec3(c.z, c.y, c.x);
        }

        vec4 getA(ivec3 coords) 
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        void main() 
        {
            ivec3 coords = getOutCoords();

            ivec4 v = toInt(getA(coords)); // -2048..2047 half float precision 
            uvec4 u = uvec4(v + ivec4(2048)); // 0..4095
            uvec4 o = (u >> map) & 1u;

            setOutput(vec4(o));
        }
        `
    }
}

export function computeUnidirectionalOcclusionMap(volume: tf.Tensor3D, permute: Permute, reverse: Reverse) : tf.Tensor3D
{
    const reversed = volume.reverse(reverse) as tf.Tensor3D
    const transposed = reversed.transpose(permute) as tf.Tensor3D
    reversed.dispose()

    const minimaProgram = new GPGPUUnidirectionalMinimaMap(transposed.shape)
    const minimaRaw = runWebGLProgram(minimaProgram, [transposed], 'float32', [], true) 
    // logTensor('minimaRaw', minimaRaw)

    const slices = unstackPacked(minimaRaw, 0) 
    minimaRaw.dispose()

    const sliceShape = slices[0].shape as [number, number, number, 2, 2]
    const updateProgram = new GPGPUUpdateUnidirectionalMinimaSlices(sliceShape)

    for (let i = 1; i < slices.length; i++)
    {
        const updatedSlice = runWebGLProgram(updateProgram, [slices[i], slices[i-1]], 'float32', [], true)
        tf.dispose(slices[i])
        slices[i] = updatedSlice
    }

    const minima = stackPacked(slices, 0) 
    // logTensor('minima', minima)
    tf.dispose(slices)
    
    const maximaProgram = new GPGPUUnidirectionalMaximaMap(transposed.shape)
    const maxima = runWebGLProgram(maximaProgram, [transposed], 'float32', [], true)
    // logTensor('maxima', maxima)
    tf.dispose(transposed)

    const occlusionShape = minima.shape.slice(0, 3) as [number, number, number]
    const occlusionProgram = new GPGPUUnidirectionalOcclusionMap(occlusionShape)
    const occlusion = runWebGLProgram(occlusionProgram, [minima, maxima], 'float32', [], true) as tf.Tensor3D
    // logTensor('occlusion', occlusion)
    tf.dispose([minima, maxima])

    const untransposed = occlusion.transpose(inversePermute(permute))
    tf.dispose(occlusion)

    const unreversed = untransposed.reverse(reverse)
    tf.dispose(untransposed)

    return unreversed as tf.Tensor3D
}

export function computeBidirectionalOcclusionMap(volumeMap: tf.Tensor3D, permute: Permute, reverse: Reverse) : tf.Tensor3D
{
    const occlusionMaps = [
        computeUnidirectionalOcclusionMap(volumeMap, permute, reverse),
        computeUnidirectionalOcclusionMap(volumeMap, permute, complementReverse(reverse)),
    ]

    const bidirectionalProgram = new GPGPUBidirectionalOcclusionMap(occlusionMaps[0].shape)
    const occlusionMap = runWebGLProgram(bidirectionalProgram, occlusionMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)
    // logTensor('bidirectionalOcclusion', occlusionMap)

    return occlusionMap as tf.Tensor3D
}

export function computeAnisotropicBidirectionalOcclusionMap(volumeMap: tf.Tensor3D, permute: Permute) : tf.Tensor3D
{
    const reverseA = permute.slice(1, 1) as Reverse
    const reverseB = permute.slice(1, 2) as Reverse
    const reverseC = permute.slice(2, 3) as Reverse
    const reverseD = permute.slice(1, 3) as Reverse

    const occlusionMaps = [
        computeBidirectionalOcclusionMap(volumeMap, permute, reverseA),
        computeBidirectionalOcclusionMap(volumeMap, permute, reverseB),
        computeBidirectionalOcclusionMap(volumeMap, permute, reverseC),
        computeBidirectionalOcclusionMap(volumeMap, permute, reverseD),
    ]

    const anisotropicBidirectionalProgram = new GPGPUAnisotropicBidirectionalOcclusionMap(occlusionMaps[0].shape)
    const occlusionMap = runWebGLProgram(anisotropicBidirectionalProgram, occlusionMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)
    // logAnisotropicBidirectionalOcclusionMaps(occlusionMap)

    return occlusionMap 
}

export function computeExtendedAnisotropicBidirectionalOcclusionMap(volumeMap: tf.Tensor3D) : tf.Tensor3D
{
    const permuteX = [2,1,0] as Permute
    const permuteY = [1,2,0] as Permute
    const permuteZ = [0,1,2] as Permute

    const occlusionMaps = [
        computeAnisotropicBidirectionalOcclusionMap(volumeMap, permuteX),
        computeAnisotropicBidirectionalOcclusionMap(volumeMap, permuteY),
        computeAnisotropicBidirectionalOcclusionMap(volumeMap, permuteZ),
    ]

    const extendedAnisotropicBidirectionalProgram = new GPGPUExtendedAnisotropicBidirectionalOcclusionMap(occlusionMaps[0].shape)
    const occlusionMap = runWebGLProgram(extendedAnisotropicBidirectionalProgram, occlusionMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)
    // logExtendedAnisotropicBidirectionalOcclusionMaps(occlusionMap)

    return occlusionMap 
}

// async functions

export async function computeUnidirectionalOcclusionMapAsync(volume: tf.Tensor3D, permute: Permute, reverse: Reverse) : Promise<tf.Tensor3D>
{
    const reversed = volume.reverse(reverse) as tf.Tensor3D
    const transposed = reversed.transpose(permute) as tf.Tensor3D
    reversed.dispose()

    const minimaProgram = new GPGPUUnidirectionalMinimaMap(transposed.shape)
    let minima = runWebGLProgram(minimaProgram, [transposed], 'float32', [], true)
    // logTensor('minimaRaw', minima) 

    const minimaShape = minima.shape as [number, number, number, 2, 2]
    const updateProgram = new GPGPUUpdateUnidirectionalMinimaMap(minimaShape)

    for (let i = 1; i < minimaShape[0]; i++)
    {
        const t = runWebGLProgram(updateProgram, [minima], 'float32', [], true)
        tf.dispose(minima)
        minima = t

        await tf.nextFrame()
    }
    // logTensor('minima', minima) 

    const maximaProgram = new GPGPUUnidirectionalMaximaMap(transposed.shape)
    const maxima = runWebGLProgram(maximaProgram, [transposed], 'float32', [], true)
    // logTensor('maxima', maxima) 
    tf.dispose(transposed)

    const occlusionShape = minimaShape.slice(0,3) as [number, number, number]
    const occlusionProgram = new GPGPUUnidirectionalOcclusionMap(occlusionShape)
    const occlusion = runWebGLProgram(occlusionProgram, [minima, maxima], 'float32', [], true)
    tf.dispose([minima, maxima])
    // logTensor('occlusion', occlusion) 

    const untransposed = occlusion.transpose(inversePermute(permute))
    tf.dispose(occlusion)

    const unreversed = untransposed.reverse(reverse)
    tf.dispose(untransposed)

    return unreversed as tf.Tensor3D
}

export async function computeBidirectionalOcclusionMapAsync(volumeMap: tf.Tensor3D, permute: Permute, reverse: Reverse): Promise<tf.Tensor3D>
{
    const occlusionMaps = [
        await computeUnidirectionalOcclusionMapAsync(volumeMap, permute, reverse),
        await computeUnidirectionalOcclusionMapAsync(volumeMap, permute, complementReverse(reverse)),
    ]

    const bidirectionalProgram = new GPGPUBidirectionalOcclusionMap(occlusionMaps[0].shape)
    const occlusionMap = runWebGLProgram(bidirectionalProgram, occlusionMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)
    // logTensor('bidirectionalOcclusion', occlusionMap)

    return occlusionMap as tf.Tensor3D
}

export async function computeAnisotropicBidirectionalOcclusionMapAsync(volumeMap: tf.Tensor3D, permute: Permute) : Promise<tf.Tensor3D>
{
    const reverseA = permute.slice(1, 1) as Reverse
    const reverseB = permute.slice(1, 2) as Reverse
    const reverseC = permute.slice(2, 3) as Reverse
    const reverseD = permute.slice(1, 3) as Reverse

    const occlusionMaps = [
        await computeBidirectionalOcclusionMapAsync(volumeMap, permute, reverseA),
        await computeBidirectionalOcclusionMapAsync(volumeMap, permute, reverseB),
        await computeBidirectionalOcclusionMapAsync(volumeMap, permute, reverseC),
        await computeBidirectionalOcclusionMapAsync(volumeMap, permute, reverseD),
    ]

    const anisotropicBidirectionalProgram = new GPGPUAnisotropicBidirectionalOcclusionMap(occlusionMaps[0].shape)
    const occlusionMap = runWebGLProgram(anisotropicBidirectionalProgram, occlusionMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)
    // logAnisotropicBidirectionalOcclusionMaps(occlusionMap)

    return occlusionMap 
}

export async function computeExtendedAnisotropicBidirectionalOcclusionMapAsync(volumeMap: tf.Tensor3D): Promise<tf.Tensor3D>
{
    const permuteX = [2,1,0] as Permute
    const permuteY = [1,2,0] as Permute
    const permuteZ = [0,1,2] as Permute

    const occlusionMaps = [
        await computeAnisotropicBidirectionalOcclusionMapAsync(volumeMap, permuteX),
        await computeAnisotropicBidirectionalOcclusionMapAsync(volumeMap, permuteY),
        await computeAnisotropicBidirectionalOcclusionMapAsync(volumeMap, permuteZ),
    ]

    const extendedAnisotropicBidirectionalProgram = new GPGPUExtendedAnisotropicBidirectionalOcclusionMap(occlusionMaps[0].shape)
    const occlusionMap = runWebGLProgram(extendedAnisotropicBidirectionalProgram, occlusionMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)
    // logExtendedAnisotropicBidirectionalOcclusionMaps(occlusionMap)

    return occlusionMap 
}

// helper function

function logTensor(name: string, tensor: tf.Tensor)
{
    console.log(name, tf.tidy(() => tensor.mean([0,1,2]).dataSync())) 
}

function logAnisotropicBidirectionalOcclusionMaps(occlusionMaps: tf.Tensor3D)
{
    const unpack = new GPGPUUnpackFromAnisotropicBidirectionalOcclusionMap(occlusionMaps.shape)

    console.log('occlusionMap0', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[0]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMap1', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[1]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMap2', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[2]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMap3', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[3]]).mean([0,1,2]).dataSync())) 
}

function logExtendedAnisotropicBidirectionalOcclusionMaps(occlusionMaps: tf.Tensor3D)
{
    const unpack = new GPGPUUnpackFromExtendedAnisotropicBidirectionalOcclusionMap(occlusionMaps.shape)

    console.log('occlusionMapX0', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[ 0]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMapX1', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[ 1]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMapX2', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[ 2]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMapX3', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[ 3]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMapY0', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[ 4]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMapY1', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[ 5]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMapY2', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[ 6]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMapY3', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[ 7]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMapZ0', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[ 8]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMapZ1', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[ 9]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMapZ2', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[10]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMapZ3', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[11]]).mean([0,1,2]).dataSync())) 
}

function runWebGLProgram(
    prog: GPGPUProgram, 
    inputs: tf.Tensor[], 
    outputDtype: tf.DataType = 'float32', 
    customUniformValues: number[][] = [], 
    preventEagerUnpackingOfOutput: boolean = true
): tf.Tensor
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(prog, inputs, outputDtype, customUniformValues, preventEagerUnpackingOfOutput)
    return tf.engine().makeTensorFromTensorInfo(info) as tf.Tensor
}
