import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { unstackPacked } from './unstack_packed_keepDims_webgl'
import { stackPacked } from './stack_packed_keepDims_webgl'

type Axis = 0 | 1 | 2
type Permute = [Axis, Axis, Axis]
type Reverse = Axis[]

class UnidirectionalMinimaMapDeprecated implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(inputShape: [number, number, number], permutation: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [inDepth, inHeight, inWidth] = inputShape
        const [outDepth, outHeight, outWidth] = inputShape.map(x => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]  

        const transformVoxelOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permutation)

            for (const a of reverse) old[a] = 1 - old[a]
            
            return old.toReversed().join(',')
        }
    
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

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
            ivec5 cCoords = getOutputCoords();
            return ivec3(cCoords.z, cCoords.y, cCoords.x);
        }

        ivec3 getVCoords(ivec3 vCoords, int ox, int oy, int oz)
        {
            return vCoords + ivec3(ox, oy, oz);
        }

        float getA(ivec3 vCoords)
        {
            vCoords = clamp(vCoords, minCoords, maxCoords);
            return getA(vCoords.z, vCoords.y, vCoords.x);
        }

        CellValues getValues(ivec3 cCoords)
        {
            CellValues c;

            ivec3 vCoords = cCoords - 1;

            c.v000 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,0,0)}));
            c.v100 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,0,0)}));
            c.v010 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,1,0)}));
            c.v001 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,0,1)}));
            c.v011 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,1,1)}));
            c.v101 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,0,1)}));
            c.v110 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,1,0)}));
            c.v111 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,1,1)}));

            return c;
        }

        float getMinOnFaceX(CellValues c)
        {           
            return min4(c.v100, c.v110, c.v101, c.v111);
        }

        float getMinOnFaceY(CellValues c)
        {
            return min4(c.v010, c.v110, c.v011, c.v111);
        }
            
        float getMinOnFaceZ(CellValues c)
        {
            return min4(c.v001, c.v011, c.v101, c.v111);
        }

        void main()
        {
            ivec3 cCoords = getOutCoords();
            CellValues c = getValues(cCoords);

            float xMin = getMinOnFaceX(c);
            float yMin = getMinOnFaceY(c);
            float zMin = getMinOnFaceZ(c);

            setOutput(vec4(xMin, yMin, zMin, 0.0));
        }
        `
    }
}

class UnidirectionalMinimaMapDeprecated2 implements GPGPUProgram 
{
    variableNames: string[]
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(inputShape: [number, number, number], permutation: Permute = [0,1,2], reverse: Reverse = [], shadowed: boolean = false) 
    {
        this.variableNames = shadowed ? ['A', 'B'] : ['A'];

        const [inDepth, inHeight, inWidth] = inputShape
        const [outDepth, outHeight, outWidth] = inputShape.map(x => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]  

        const transformVoxelOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permutation)

            for (const a of reverse) old[a] = 1 - old[a]

            return old.toReversed().join(',')
        }

        const getB = () => shadowed ? `
        const float negInf = uintBitsToFloat(0xff800000u);

        const ivec3 minCCoords = ivec3(0);
        const ivec3 maxCCoords = ivec3(${outDepth-1}, ${outHeight-1}, ${outWidth-1});

        float getB(ivec3 cCoords)
        {
            cCoords = clamp(cCoords, minCCoords, maxCCoords);
            return getB(cCoords.z, cCoords.y, cCoords.x);
        }` : ``

        const returnEarly = () => shadowed ? `
        if (getB(cCoords) > 0.5)
        {
            setOutput(vec4(vec3(negInf), 1.0));
            return;
        }
        ` : ``

        this.userCode = `
        const ivec3 minVCoords = ivec3(0);
        const ivec3 maxVCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

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
            ivec5 cCoords = getOutputCoords();
            return ivec3(cCoords.z, cCoords.y, cCoords.x);
        }

        ivec3 getVCoords(ivec3 vCoords, int ox, int oy, int oz)
        {
            return vCoords + ivec3(ox, oy, oz);
        }

        float getA(ivec3 vCoords)
        {
            vCoords = clamp(vCoords, minVCoords, maxVCoords);
            return getA(vCoords.z, vCoords.y, vCoords.x);
        }

        ${getB()}

        CellValues getValues(ivec3 cCoords)
        {
            CellValues c;

            ivec3 vCoords = cCoords - 1;

            c.v000 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,0,0)}));
            c.v100 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,0,0)}));
            c.v010 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,1,0)}));
            c.v001 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,0,1)}));
            c.v011 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,1,1)}));
            c.v101 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,0,1)}));
            c.v110 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,1,0)}));
            c.v111 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,1,1)}));

            return c;
        }

        float getMinOnFaceX(CellValues c)
        {           
            return min4(c.v100, c.v110, c.v101, c.v111);
        }

        float getMinOnFaceY(CellValues c)
        {
            return min4(c.v010, c.v110, c.v011, c.v111);
        }
            
        float getMinOnFaceZ(CellValues c)
        {
            return min4(c.v001, c.v011, c.v101, c.v111);
        }

        void main()
        {
            ivec3 cCoords = getOutCoords();
           
            ${returnEarly()}

            CellValues c = getValues(cCoords);

            float xMin = getMinOnFaceX(c);
            float yMin = getMinOnFaceY(c);
            float zMin = getMinOnFaceZ(c);

            setOutput(vec4(xMin, yMin, zMin, 1.0));
            
        }
        `
    }
}

class UnidirectionalMaximaMapDeprecated implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(inputShape: [number, number, number], permutation: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [inDepth, inHeight, inWidth] = inputShape
        const [outDepth, outHeight, outWidth] = inputShape.map(x => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]     

        const transformVoxelOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permutation)

            for (const a of reverse) old[a] = 1 - old[a]
            
            return old.toReversed().join(',')
        }

        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

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
            ivec5 cCoords = getOutputCoords();
            return ivec3(cCoords.z, cCoords.y, cCoords.x);
        }

        ivec3 getVCoords(ivec3 vCoords, int ox, int oy, int oz)
        {
            return vCoords + ivec3(ox, oy, oz);
        }

        float getA(ivec3 vCoords)
        {
            vCoords = clamp(vCoords, minCoords, maxCoords);
            return getA(vCoords.z, vCoords.y, vCoords.x);
        }

        CellValues getValues(ivec3 cCoords)
        {
            CellValues c;

            ivec3 vCoords = cCoords - 1;

            c.v000 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,0,0)}));
            c.v100 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,0,0)}));
            c.v010 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,1,0)}));
            c.v001 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,0,1)}));
            c.v011 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,1,1)}));
            c.v101 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,0,1)}));
            c.v110 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,1,0)}));
            c.v111 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,1,1)}));

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
            ivec3 cCoords = getOutCoords();
            CellValues c = getValues(cCoords);

            float xMax = getMaxOnFaceX(c);
            float yMax = getMaxOnFaceY(c);
            float zMax = getMaxOnFaceZ(c);

            setOutput(vec4(xMax, yMax, zMax, 1.0));
        }
        `
    }
}

class UnidirectionalMinimaMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(inputShape: [number, number, number], permutation: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [inDepth, inHeight, inWidth] = inputShape
        const [outDepth, outHeight, outWidth] = inputShape.map(x => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]   
        
        const transformVoxelOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permutation)

            for (const a of reverse) old[a] = 1 - old[a]
            
            return old.toReversed().join(',')
        }

        const transformCellOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permutation)

            for (const a of reverse) old[a] = - old[a]
            
            return old.toReversed().join(',')
        }

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
            ivec5 cCoords = getOutputCoords();
            return ivec3(cCoords.z, cCoords.y, cCoords.x);
        }

        ivec3 getVCoords(ivec3 vCoords, int ox, int oy, int oz)
        {
            return vCoords + ivec3(ox, oy, oz);
        }

        ivec3 getCCoords(ivec3 cCoords, int ox, int oy, int oz)
        {
            return cCoords + ivec3(ox, oy, oz);
        }

        float getA(ivec3 vCoords)
        {
            vCoords = clamp(vCoords, minCoords, maxCoords);
            return getA(vCoords.z, vCoords.y, vCoords.x);
        }

        CellValues getValues(ivec3 cCoords)
        {
            CellValues c;

            ivec3 vCoords = cCoords - 1;

            c.v000 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,0,0)}));
            c.v100 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,0,0)}));
            c.v010 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,1,0)}));
            c.v001 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,0,1)}));
            c.v011 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,1,1)}));
            c.v101 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,0,1)}));
            c.v110 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,1,0)}));
            c.v111 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,1,1)}));

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
            ivec3 cCoords = getOutCoords();

            CellValues c000 = getValues(getCCoords(cCoords, ${transformCellOffset(0,0,0)}));
            CellValues c100 = getValues(getCCoords(cCoords, ${transformCellOffset(1,0,0)}));
            CellValues c010 = getValues(getCCoords(cCoords, ${transformCellOffset(0,1,0)}));
            CellValues c001 = getValues(getCCoords(cCoords, ${transformCellOffset(0,0,1)}));

            float xMin = getMinOnFaceX(c000, c100);
            float yMin = getMinOnFaceY(c000, c010);
            float zMin = getMinOnFaceZ(c000, c001);

            setOutput(vec4(xMin, yMin, zMin, 0.0));
        }
        `
    }
}

class UpdateUnidirectionalMinimaSlices implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(outputShape: [number, number, number, 2, 2], permutation: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [outDepth, outHeight, outWidth] = outputShape.slice(0, 3)
        this.outputShape = outputShape 

        const transformCellOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permutation)

            for (const a of reverse) old[a] = - old[a]
            
            return old.toReversed().join(',')
        }

        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        float min3(float a, float b, float c) { return min(min(a, b), c); }

        ivec3 getOutCoords()
        {
            ivec5 cCoords = getOutputCoords();
            return ivec3(cCoords.z, cCoords.y, cCoords.x);
        }

        ivec3 getCCoords(ivec3 cCoords, int ox, int oy, int oz)
        {
            return cCoords + ivec3(ox, oy, oz);
        }

        vec4 getA(ivec3 cCoords)
        {
            cCoords = clamp(cCoords, minCoords, maxCoords);
            return getA(cCoords.z, cCoords.y, cCoords.x, 0, 0);
        }

        vec4 getB(ivec3 cCoords)
        {
            cCoords = clamp(cCoords, minCoords, maxCoords);
            return getB(cCoords.z, cCoords.y, cCoords.x, 0, 0);
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
            ivec3 cCoords = getOutCoords();

            vec4 c111 = getA(getCCoords(cCoords, ${transformCellOffset(-0,-0,-0)}));
            vec4 c011 = getA(getCCoords(cCoords, ${transformCellOffset(-1,-0,-0)}));
            vec4 c101 = getA(getCCoords(cCoords, ${transformCellOffset(-0,-1,-0)}));
            vec4 c001 = getA(getCCoords(cCoords, ${transformCellOffset(-1,-1,-0)}));
            vec4 c110 = getB(getCCoords(cCoords, ${transformCellOffset(-0,-0,-1)}));
            vec4 c010 = getB(getCCoords(cCoords, ${transformCellOffset(-1,-0,-1)}));
            vec4 c100 = getB(getCCoords(cCoords, ${transformCellOffset(-0,-1,-1)}));
            vec4 c000 = getB(getCCoords(cCoords, ${transformCellOffset(-1,-1,-1)}));

            c111.x = getMinOnFaceX(c111, c110, c101, c100);
            c111.y = getMinOnFaceY(c111, c110, c011, c010);
            c111.z = getMinOnFaceZ(c111, c110, c101, c011, c100, c010, c001, c000);

            setOutput(c111);
        }
        `
    }
}

class UpdateUnidirectionalMinimaMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(outputShape: [number, number, number, 2, 2], permutation: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [outDepth, outHeight, outWidth] = outputShape.slice(0,3)
        this.outputShape = outputShape  

        const transformCellOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permutation)

            for (const a of reverse) old[a] = - old[a]
            
            return old.toReversed().join(',')
        }

        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        float min3(float a, float b, float c) { return min(min(a, b), c); }

        ivec3 getOutCoords()
        {
            ivec5 cCoords = getOutputCoords();
            return ivec3(cCoords.z, cCoords.y, cCoords.x);
        }

        ivec3 getCCoords(ivec3 cCoords, int ox, int oy, int oz)
        {
            return cCoords + ivec3(ox, oy, oz);
        }

        vec4 getA(ivec3 cCoords)
        {
            cCoords = clamp(cCoords, minCoords, maxCoords);
            return getA(cCoords.z, cCoords.y, cCoords.x, 0, 0);
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
            ivec3 cCoords = getOutCoords();

            vec4 c111 = getA(getCCoords(cCoords, ${transformCellOffset(-0,-0,-0)}));
            vec4 c011 = getA(getCCoords(cCoords, ${transformCellOffset(-1,-0,-0)}));
            vec4 c101 = getA(getCCoords(cCoords, ${transformCellOffset(-0,-1,-0)}));
            vec4 c001 = getA(getCCoords(cCoords, ${transformCellOffset(-1,-1,-0)}));
            vec4 c110 = getA(getCCoords(cCoords, ${transformCellOffset(-0,-0,-1)}));
            vec4 c010 = getA(getCCoords(cCoords, ${transformCellOffset(-1,-0,-1)}));
            vec4 c100 = getA(getCCoords(cCoords, ${transformCellOffset(-0,-1,-1)}));
            vec4 c000 = getA(getCCoords(cCoords, ${transformCellOffset(-1,-1,-1)}));

            c111.x = getMinOnFaceX(c111, c110, c101, c100);
            c111.y = getMinOnFaceY(c111, c110, c011, c010);
            c111.z = getMinOnFaceZ(c111, c110, c101, c011, c100, c010, c001, c000);

            setOutput(c111);
        }
        `
    }
}

class UnidirectionalMaximaMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(inputShape: [number, number, number], permutation: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [inDepth, inHeight, inWidth] = inputShape
        const [outDepth, outHeight, outWidth] = inputShape.map(x => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]   

        const transformVoxelOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permutation)

            for (const a of reverse) old[a] = 1 - old[a]
            
            return old.toReversed().join(',')
        }

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
            ivec5 cCoords = getOutputCoords();
            return ivec3(cCoords.z, cCoords.y, cCoords.x);
        }

        ivec3 getVCoords(ivec3 vCoords, int ox, int oy, int oz)
        {
            return vCoords + ivec3(ox, oy, oz);
        }

        float getA(ivec3 vCoords)
        {
            vCoords = clamp(vCoords, minCoords, maxCoords);
            return getA(vCoords.z, vCoords.y, vCoords.x);
        }

        CellValues getValues(ivec3 cCoords)
        {
            CellValues c;

            ivec3 vCoords = cCoords - 1;

            c.v000 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,0,0)}));
            c.v100 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,0,0)}));
            c.v010 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,1,0)}));
            c.v001 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,0,1)}));
            c.v011 = getA(getVCoords(vCoords, ${transformVoxelOffset(0,1,1)}));
            c.v101 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,0,1)}));
            c.v110 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,1,0)}));
            c.v111 = getA(getVCoords(vCoords, ${transformVoxelOffset(1,1,1)}));

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

        float getMaxDiffOnCell(CellValues c)
        {
            float m = -1.0;

            m = max(m, avg3(c.v001, c.v010, c.v100) - c.v000); 
            m = max(m, avg3(c.v001, c.v010, c.v011) - c.v000); 
            m = max(m, avg3(c.v001, c.v100, c.v101) - c.v000); 
            m = max(m, avg3(c.v011, c.v101, c.v110) - c.v000); 
            m = max(m, avg3(c.v101, c.v110, c.v111) - c.v100); 
            m = max(m, avg3(c.v011, c.v110, c.v111) - c.v010); 
            m = max(m, avg3(c.v000, c.v001, c.v010) - c.v000); 
            m = max(m, avg3(c.v000, c.v001, c.v100) - c.v000); 
            m = max(m, avg3(c.v100, c.v101, c.v110) - c.v100); 
            m = max(m, avg3(c.v010, c.v011, c.v110) - c.v010); 
            m = max(m, c.v001 - c.v000); 
            m = max(m, c.v011 - c.v000); 
            m = max(m, c.v101 - c.v000); 
            m = max(m, c.v111 - c.v000); 
            m = max(m, c.v101 - c.v100); 
            m = max(m, c.v111 - c.v100); 
            m = max(m, c.v011 - c.v010); 
            m = max(m, c.v111 - c.v010); 
            m = max(m, c.v111 - c.v110); 

            return m;
        }

        void main()
        {
            ivec3 cCoords = getOutCoords();
            CellValues c = getValues(cCoords);

            float xMax = getMaxOnFaceX(c);
            float yMax = getMaxOnFaceY(c);
            float zMax = getMaxOnFaceZ(c);
            float wMax = getMaxDiffOnCell(c);

            setOutput(vec4(xMax, yMax, zMax, wMax));
        }
        `
    }
}

class UnidirectionalShadowMap implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = false

    constructor(outputShape: [number, number, number], permutation: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [outDepth, outHeight, outWidth] = outputShape
        this.outputShape = outputShape

        const transformCellOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permutation)

            for (const a of reverse) old[a] = - old[a]
            
            return old.toReversed().join(',')
        }

        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        ivec3 getOutCoords()
        {
            ivec3 cCoords = getOutputCoords();
            return ivec3(cCoords.z, cCoords.y, cCoords.x);
        }

        ivec3 getCCoords(ivec3 cCoords, int ox, int oy, int oz)
        {
            return cCoords + ivec3(ox, oy, oz);
        }

        vec4 getA(ivec3 cCoords)
        {
            cCoords = clamp(cCoords, minCoords, maxCoords);
            return getA(cCoords.z, cCoords.y, cCoords.x, 0, 0);
        }

        vec4 getB(ivec3 cCoords)
        {
            cCoords = clamp(cCoords, minCoords, maxCoords);
            return getB(cCoords.z, cCoords.y, cCoords.x, 0, 0);
        }

        bool isShadowed(vec4 minValues, vec4 maxValues)
        {
            bvec4 tests = greaterThanEqual(minValues, maxValues);
            bool shadowed = all(tests.xyz) || tests.w;

            return shadowed;
        }
                
        void main()
        {
            ivec3 cCoords = getOutCoords();

            vec4 c111 = getB(getCCoords(cCoords, ${transformCellOffset(-0,-0,-0)}));
            vec4 c011 = getA(getCCoords(cCoords, ${transformCellOffset(-1,-0,-0)}));
            vec4 c101 = getA(getCCoords(cCoords, ${transformCellOffset(-0,-1,-0)}));
            vec4 c110 = getA(getCCoords(cCoords, ${transformCellOffset(-0,-0,-1)}));

            vec4 minValues = vec4(c011.x, c101.y, c110.z, 0.0);
            vec4 maxValues = vec4(c111.x, c111.y, c111.z, c111.w);

            setOutput(float(isShadowed(minValues, maxValues)));
        }
        `
    }
}

class BidirectionalShadowMap implements GPGPUProgram 
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

            uvec4 sA = toUint(getA(coords));
            uvec4 sB = toUint(getB(coords));

            setOutput(vec4(sA | sB));
        }
        `
    }
}

class AnisotropicBidirectionalShadowMap implements GPGPUProgram 
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

        uvec4 bitpack(uvec4 sA, uvec4 sB, uvec4 sC, uvec4 sD) 
        { 
            return (sA << 0) | (sB << 1) | (sC << 2) | (sD << 3);
        }
 
        void main()
        {
            ivec3 coords = getOutCoords();

            uvec4 sA = toUint(getA(coords));
            uvec4 sB = toUint(getB(coords));
            uvec4 sC = toUint(getC(coords));
            uvec4 sD = toUint(getD(coords));

            setOutput(vec4(bitpack(sA, sB, sC, sD)));
        }
        `
    }
}

class ExtendedAnisotropicBidirectionalShadowMap implements GPGPUProgram 
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

        ivec4 bitpack(uvec4 sA, uvec4 sB, uvec4 sC)
        {
            uvec4 p = (sA << 0u) | (sB << 4u) | (sC << 8u); // 0..4095
            return ivec4(p) - ivec4(2048); // -2048..2047 in half float precision 
        }
 
        void main()
        {
            ivec3 coords = getOutCoords();

            uvec4 sA = toUint(getA(coords));
            uvec4 sB = toUint(getB(coords));
            uvec4 sC = toUint(getC(coords));

            setOutput(vec4(bitpack(sA, sB, sC)));
        }
        `
    }
}

class UnpackAnisotropicBidirectionalShadowMap implements GPGPUProgram 
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
            uvec4 s = (u >> map) & 1u;

            setOutput(vec4(s));
        }
        `
    }
}

class UnpackExtendedAnisotropicBidirectionalShadowMap implements GPGPUProgram 
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
            uvec4 s = (u >> map) & 1u;

            setOutput(vec4(s));
        }
        `
    }
}

// sync functions 

export function computeUnidirectionalShadowMap(volume: tf.Tensor3D, permutation: Permute, reverse: Reverse, verbose: boolean = false) : tf.Tensor3D
{
    const axis = permutation[0]
    const reverseAxis = reverse.includes(axis)

    const minimaProgram = new UnidirectionalMinimaMap(volume.shape, permutation, reverse)
    const minimaStack = runWebGLProgram(minimaProgram, [volume], 'float32', [], true)
    if (verbose) logTensor('minimaStack', minimaStack)

    const slices = unstackPacked(minimaStack, axis) 
    minimaStack.dispose()
    if (reverseAxis) slices.reverse()

    const sliceShape = slices[0].shape as [number, number, number, 2, 2]
    const sliceProgram = new UpdateUnidirectionalMinimaSlices(sliceShape, permutation, reverse)

    for (let i = 1; i < slices.length; i++)
    {
        const t = runWebGLProgram(sliceProgram, [slices[i], slices[i-1]], 'float32', [[i]], true)
        tf.dispose(slices[i])
        slices[i] = t
    }

    if (reverseAxis) slices.reverse()
    const minima = stackPacked(slices, axis) 
    tf.dispose(slices)
    if (verbose) logTensor('minima', minima)

    const maximaProgram = new UnidirectionalMaximaMap(volume.shape, permutation, reverse)
    const maxima = runWebGLProgram(maximaProgram, [volume], 'float32', [], true)
    if (verbose) logTensor('maxima', maxima)

    const shadowShape = minima.shape.slice(0,3) as [number, number, number]
    const shadowProgram = new UnidirectionalShadowMap(shadowShape, permutation, reverse)
    const shadow = runWebGLProgram(shadowProgram, [minima, maxima], 'float32', [], true) as tf.Tensor3D
    tf.dispose([minima, maxima])
    if (verbose) logTensor('shadow', shadow)

    return shadow as tf.Tensor3D
}

export function computeBidirectionalShadowMap(volumeMap: tf.Tensor3D, permutation: Permute, reverse: Reverse, verbose: boolean = false) : tf.Tensor3D
{
    const shadowMaps = [
        computeUnidirectionalShadowMap(volumeMap, permutation, reverse),
        computeUnidirectionalShadowMap(volumeMap, permutation, complementReverse(reverse)),
    ]

    const bidirectionalProgram = new BidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(bidirectionalProgram, shadowMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logTensor('bidirectionalShadow', shadowMap)

    return shadowMap as tf.Tensor3D
}

export function computeAnisotropicBidirectionalShadowMap(volumeMap: tf.Tensor3D, permutation: Permute, verbose: boolean = false) : tf.Tensor3D
{
    const reversePP = permutation.slice(1, 1) as Reverse
    const reversePN = permutation.slice(1, 2) as Reverse
    const reverseNP = permutation.slice(2, 3) as Reverse
    const reverseNN = permutation.slice(1, 3) as Reverse

    const shadowMaps = [
        computeBidirectionalShadowMap(volumeMap, permutation, reversePP),
        computeBidirectionalShadowMap(volumeMap, permutation, reversePN),
        computeBidirectionalShadowMap(volumeMap, permutation, reverseNP),
        computeBidirectionalShadowMap(volumeMap, permutation, reverseNN),
    ]

    const anisotropicBidirectionalProgram = new AnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(anisotropicBidirectionalProgram, shadowMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logAnisotropicBidirectionalShadowMaps(shadowMap)

    return shadowMap 
}

export function computeExtendedAnisotropicBidirectionalShadowMap(volumeMap: tf.Tensor3D, verbose: boolean = false) : tf.Tensor3D
{
    const permuteX = [2,1,0] as Permute
    const permuteY = [1,2,0] as Permute
    const permuteZ = [0,1,2] as Permute

    const shadowMaps = [
        computeAnisotropicBidirectionalShadowMap(volumeMap, permuteX),
        computeAnisotropicBidirectionalShadowMap(volumeMap, permuteY),
        computeAnisotropicBidirectionalShadowMap(volumeMap, permuteZ),
    ]

    const extendedAnisotropicBidirectionalProgram = new ExtendedAnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(extendedAnisotropicBidirectionalProgram, shadowMaps, 'int32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logExtendedAnisotropicBidirectionalShadowMaps(shadowMap)

    return shadowMap 
}

// async functions

export async function computeUnidirectionalShadowMapAsync(volume: tf.Tensor3D, permutation: Permute, reverse: Reverse, verbose: boolean = false) : Promise<tf.Tensor3D>
{
    const minimaProgram = new UnidirectionalMinimaMap(volume.shape, permutation, reverse)
    let minima = runWebGLProgram(minimaProgram, [volume], 'float32', [], true)
    if (verbose) logTensor('minimaRaw', minima)

    const minimaShape = minima.shape as [number, number, number, 2, 2]
    const updateProgram = new UpdateUnidirectionalMinimaMap(minimaShape, permutation, reverse)

    for (let i = 1; i < minimaShape[permutation[0]]; i++)
    {
        const map = runWebGLProgram(updateProgram, [minima], 'float32', [], true)
        tf.dispose(minima)
        minima = map

        await tf.nextFrame()
    }
    if (verbose) logTensor('minima', minima)

    const maximaProgram = new UnidirectionalMaximaMap(volume.shape, permutation, reverse)
    const maxima = runWebGLProgram(maximaProgram, [volume], 'float32', [], true)
    if (verbose) logTensor('maxima', maxima)

    const shadowShape = minimaShape.slice(0,3) as [number, number, number]
    const shadowProgram = new UnidirectionalShadowMap(shadowShape, permutation, reverse)
    const shadow = runWebGLProgram(shadowProgram, [minima, maxima], 'float32', [], true)
    tf.dispose([minima, maxima])
    if (verbose) logTensor('shadow', shadow)

    return shadow as tf.Tensor3D
}

export async function computeBidirectionalShadowMapAsync(volumeMap: tf.Tensor3D, permutation: Permute, reverse: Reverse, verbose: boolean = false) : Promise<tf.Tensor3D>
{
    const shadowMaps = [
        await computeUnidirectionalShadowMapAsync(volumeMap, permutation, reverse),
        await computeUnidirectionalShadowMapAsync(volumeMap, permutation, complementReverse(reverse)),
    ]

    const bidirectionalProgram = new BidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(bidirectionalProgram, shadowMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)
    if (verbose) logTensor('bidirectionalShadow', shadowMap)

    return shadowMap as tf.Tensor3D
}

export async function computeAnisotropicBidirectionalShadowMapAsync(volumeMap: tf.Tensor3D, permutation: Permute, verbose: boolean = false) : Promise<tf.Tensor3D>
{
    const reversePP = permutation.slice(1, 1) as Reverse
    const reversePN = permutation.slice(1, 2) as Reverse
    const reverseNP = permutation.slice(2, 3) as Reverse
    const reverseNN = permutation.slice(1, 3) as Reverse

    const shadowMaps = [
        await computeBidirectionalShadowMapAsync(volumeMap, permutation, reversePP),
        await computeBidirectionalShadowMapAsync(volumeMap, permutation, reversePN),
        await computeBidirectionalShadowMapAsync(volumeMap, permutation, reverseNP),
        await computeBidirectionalShadowMapAsync(volumeMap, permutation, reverseNN),
    ]

    const anisotropicBidirectionalProgram = new AnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(anisotropicBidirectionalProgram, shadowMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logAnisotropicBidirectionalShadowMaps(shadowMap)

    return shadowMap 
}

export async function computeExtendedAnisotropicBidirectionalShadowMapAsync(volumeMap: tf.Tensor3D, verbose: boolean = false) : Promise<tf.Tensor3D>
{
    const permuteX = [2,1,0] as Permute
    const permuteY = [1,0,2] as Permute
    const permuteZ = [0,1,2] as Permute

    const shadowMaps = [
        await computeAnisotropicBidirectionalShadowMapAsync(volumeMap, permuteX),
        await computeAnisotropicBidirectionalShadowMapAsync(volumeMap, permuteY),
        await computeAnisotropicBidirectionalShadowMapAsync(volumeMap, permuteZ),
    ]

    const extendedAnisotropicBidirectionalProgram = new ExtendedAnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(extendedAnisotropicBidirectionalProgram, shadowMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logExtendedAnisotropicBidirectionalShadowMaps(shadowMap)

    return shadowMap 
}

// deprecated functions 

export function computeUnidirectionalShadowMapDeprecated(volume: tf.Tensor3D, permutation: Permute, reverse: Reverse, verbose: boolean = false) : tf.Tensor3D
{
    const axis = permutation[0]
    const reverseAxis = reverse.includes(axis)

    const minimaProgram = new UnidirectionalMinimaMapDeprecated(volume.shape, permutation, reverse)
    const minimaStack = runWebGLProgram(minimaProgram, [volume], 'float32', [], true)
    if (verbose) logTensor('minimaStack', minimaStack)

    const slices = unstackPacked(minimaStack, axis) 
    minimaStack.dispose()
    if (reverseAxis) slices.reverse()

    const sliceShape = slices[0].shape as [number, number, number, 2, 2]
    const sliceProgram = new UpdateUnidirectionalMinimaSlices(sliceShape, permutation, reverse)

    for (let i = 1; i < slices.length; i++)
    {
        const t = runWebGLProgram(sliceProgram, [slices[i], slices[i-1]], 'float32', [[i]], true)
        tf.dispose(slices[i])
        slices[i] = t
    }

    if (reverseAxis) slices.reverse()
    const minima = stackPacked(slices, axis) 
    tf.dispose(slices)
    if (verbose) logTensor('minima', minima)

    const maximaProgram = new UnidirectionalMaximaMapDeprecated(volume.shape, permutation, reverse)
    const maxima = runWebGLProgram(maximaProgram, [volume], 'float32', [], true)
    if (verbose) logTensor('maxima', maxima)

    const shadowShape = minima.shape.slice(0,3) as [number, number, number]
    const shadowProgram = new UnidirectionalShadowMap(shadowShape, permutation, reverse)
    const shadow = runWebGLProgram(shadowProgram, [minima, maxima], 'float32', [], true) as tf.Tensor3D
    tf.dispose([minima, maxima])
    if (verbose) logTensor('shadow', shadow)

    return shadow as tf.Tensor3D
}

export function computeBidirectionalShadowMapDeprecated(volumeMap: tf.Tensor3D, permutation: Permute, reverse: Reverse, verbose: boolean = false) : tf.Tensor3D
{
    const shadowMaps = [
        computeUnidirectionalShadowMapDeprecated(volumeMap, permutation, reverse),
        computeUnidirectionalShadowMapDeprecated(volumeMap, permutation, complementReverse(reverse)),
    ]

    const bidirectionalProgram = new BidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(bidirectionalProgram, shadowMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logTensor('bidirectionalShadow', shadowMap)

    return shadowMap as tf.Tensor3D
}

export function computeAnisotropicBidirectionalShadowMapDeprecated(volumeMap: tf.Tensor3D, permutation: Permute, verbose: boolean = false) : tf.Tensor3D
{
    const reversePP = permutation.slice(1, 1) as Reverse
    const reversePN = permutation.slice(1, 2) as Reverse
    const reverseNP = permutation.slice(2, 3) as Reverse
    const reverseNN = permutation.slice(1, 3) as Reverse

    const shadowMaps = [
        computeBidirectionalShadowMapDeprecated(volumeMap, permutation, reversePP),
        computeBidirectionalShadowMapDeprecated(volumeMap, permutation, reversePN),
        computeBidirectionalShadowMapDeprecated(volumeMap, permutation, reverseNP),
        computeBidirectionalShadowMapDeprecated(volumeMap, permutation, reverseNN),
    ]

    const anisotropicBidirectionalProgram = new AnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(anisotropicBidirectionalProgram, shadowMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logAnisotropicBidirectionalShadowMaps(shadowMap)

    return shadowMap 
}

export function computeExtendedAnisotropicBidirectionalShadowMapDeprecated(volumeMap: tf.Tensor3D, verbose: boolean = false) : tf.Tensor3D
{
    const permuteX = [2,1,0] as Permute
    const permuteY = [1,2,0] as Permute
    const permuteZ = [0,1,2] as Permute

    const shadowMaps = [
        computeAnisotropicBidirectionalShadowMapDeprecated(volumeMap, permuteX),
        computeAnisotropicBidirectionalShadowMapDeprecated(volumeMap, permuteY),
        computeAnisotropicBidirectionalShadowMapDeprecated(volumeMap, permuteZ),
    ]

    const extendedAnisotropicBidirectionalProgram = new ExtendedAnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(extendedAnisotropicBidirectionalProgram, shadowMaps, 'int32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logExtendedAnisotropicBidirectionalShadowMaps(shadowMap)

    return shadowMap 
}

// ground truth

export function computeUnidirectionalShadowMapBase(volume: tf.Tensor3D, permutation: Permute, reverse: Reverse, verbose: boolean = false) : tf.Tensor3D
{
    const reversed = volume.reverse(reverse) as tf.Tensor3D
    const transposed = reversed.transpose(permutation) as tf.Tensor3D
    reversed.dispose()

    const minimaProgram = new UnidirectionalMinimaMap(transposed.shape)
    const minimaStack = runWebGLProgram(minimaProgram, [transposed], 'float32', [], true) 
    if (verbose) logTensor('minimaStack', minimaStack)

    const slices = unstackPacked(minimaStack, 0) 
    minimaStack.dispose()

    const sliceShape = slices[0].shape as [number, number, number, 2, 2]
    const updateProgram = new UpdateUnidirectionalMinimaSlices(sliceShape)

    for (let i = 1; i < slices.length; i++)
    {
        const updatedSlice = runWebGLProgram(updateProgram, [slices[i], slices[i-1]], 'float32', [], true)
        tf.dispose(slices[i])
        slices[i] = updatedSlice
    }

    const minima = stackPacked(slices, 0) 
    if (verbose) logTensor('minima', minima)
    tf.dispose(slices)
    
    const maximaProgram = new UnidirectionalMaximaMap(transposed.shape)
    const maxima = runWebGLProgram(maximaProgram, [transposed], 'float32', [], true)
    if (verbose) logTensor('maxima', maxima)
    tf.dispose(transposed)

    const shadowShape = minima.shape.slice(0, 3) as [number, number, number]
    const shadowProgram = new UnidirectionalShadowMap(shadowShape)
    const shadow = runWebGLProgram(shadowProgram, [minima, maxima], 'float32', [], true) as tf.Tensor3D
    if (verbose) logTensor('shadow', shadow)
    tf.dispose([minima, maxima])

    const untransposed = shadow.transpose(inversePermutation(permutation))
    tf.dispose(shadow)

    const unreversed = untransposed.reverse(reverse)
    tf.dispose(untransposed)

    return unreversed as tf.Tensor3D
}

export function computeBidirectionalShadowMapBase(volumeMap: tf.Tensor3D, permutation: Permute, reverse: Reverse, verbose: boolean = false) : tf.Tensor3D
{
    const shadowMaps = [
        computeUnidirectionalShadowMapBase(volumeMap, permutation, reverse),
        computeUnidirectionalShadowMapBase(volumeMap, permutation, complementReverse(reverse)),
    ]

    const bidirectionalProgram = new BidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(bidirectionalProgram, shadowMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logTensor('bidirectionalShadow', shadowMap)

    return shadowMap as tf.Tensor3D
}

export function computeAnisotropicBidirectionalShadowMapBase(volumeMap: tf.Tensor3D, permutation: Permute, verbose: boolean = false) : tf.Tensor3D
{
    const reversePP = permutation.slice(1, 1) as Reverse
    const reversePN = permutation.slice(1, 2) as Reverse
    const reverseNP = permutation.slice(2, 3) as Reverse
    const reverseNN = permutation.slice(1, 3) as Reverse

    const shadowMaps = [
        computeBidirectionalShadowMapBase(volumeMap, permutation, reversePP),
        computeBidirectionalShadowMapBase(volumeMap, permutation, reversePN),
        computeBidirectionalShadowMapBase(volumeMap, permutation, reverseNP),
        computeBidirectionalShadowMapBase(volumeMap, permutation, reverseNN),
    ]

    const anisotropicBidirectionalProgram = new AnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(anisotropicBidirectionalProgram, shadowMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logAnisotropicBidirectionalShadowMaps(shadowMap)

    return shadowMap 
}

export function computeExtendedAnisotropicBidirectionalShadowMapBase(volumeMap: tf.Tensor3D, verbose: boolean = false) : tf.Tensor3D
{
    const permuteX = [2,1,0] as Permute
    const permuteY = [1,2,0] as Permute
    const permuteZ = [0,1,2] as Permute

    const shadowMaps = [
        computeAnisotropicBidirectionalShadowMapBase(volumeMap, permuteX),
        computeAnisotropicBidirectionalShadowMapBase(volumeMap, permuteY),
        computeAnisotropicBidirectionalShadowMapBase(volumeMap, permuteZ),
    ]

    const extendedAnisotropicBidirectionalProgram = new ExtendedAnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(extendedAnisotropicBidirectionalProgram, shadowMaps, 'int32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logExtendedAnisotropicBidirectionalShadowMaps(shadowMap)

    return shadowMap 
}

// debug

export function computeExtendedAnisotropicBidirectionalShadowMapDebug(volumeMap: tf.Tensor3D, verbose: boolean = false) : tf.Tensor3D
{
    let o1,o2,o3,o4,ox,oy,oz

    // let t = computeUnidirectionalShadowMap(volumeMap, [0,1,2], [])
    // let t = computeUnidirectionalShadowMap(volumeMap, [0,1,2], [0,1,2])
    let t = computeBidirectionalShadowMapBase(volumeMap,  [2,1,0], [], true)

    o1 = tf.onesLike(t)    // computeBidirectionalShadowMap(volumeMap, [2,1,0], [   ])
    o2 = tf.onesLike(t) // computeBidirectionalShadowMap(volumeMap, [2,1,0], [  1])
    o3 = tf.onesLike(t) // computeBidirectionalShadowMap(volumeMap, [2,1,0], [  0])
    o4 = tf.onesLike(t) // computeBidirectionalShadowMap(volumeMap, [2,1,0], [1,0])

    ox = runWebGLProgram(new AnisotropicBidirectionalShadowMap(t.shape), [o1,o2,o3,o4], 'float32', [], true) as tf.Tensor3D
    tf.dispose([o1,o2,o3,o4])
   
    o1 = tf.onesLike(t) // computeBidirectionalShadowMap(volumeMap, [1,2,0], [   ])
    o2 = tf.onesLike(t) // computeBidirectionalShadowMap(volumeMap, [1,2,0], [  2])
    o3 = tf.onesLike(t) // computeBidirectionalShadowMap(volumeMap, [1,2,0], [  0])
    o4 = tf.onesLike(t) // computeBidirectionalShadowMap(volumeMap, [1,2,0], [2,0])

    oy = runWebGLProgram(new AnisotropicBidirectionalShadowMap(t.shape), [o1,o2,o3,o4], 'float32', [], true) as tf.Tensor3D
    tf.dispose([o1,o2,o3,o4])

    o1 = tf.clone(t) // computeBidirectionalShadowMap(volumeMap, [0,1,2], [   ])
    o2 = tf.onesLike(t) // computeBidirectionalShadowMap(volumeMap, [0,1,2], [  1])
    o3 = tf.onesLike(t) // computeBidirectionalShadowMap(volumeMap, [0,1,2], [  2])
    o4 = tf.onesLike(t) // computeBidirectionalShadowMap(volumeMap, [0,1,2], [1,2])

    oz = runWebGLProgram(new AnisotropicBidirectionalShadowMap(t.shape), [o1,o2,o3,o4], 'float32', [], true) as tf.Tensor3D
    tf.dispose([o1,o2,o3,o4])


    const o = runWebGLProgram(new ExtendedAnisotropicBidirectionalShadowMap(t.shape), [ox,oy,oz], 'int32', [], true) as tf.Tensor3D
    tf.dispose([ox,oy,oz])

    if (verbose) logExtendedAnisotropicBidirectionalShadowMaps(o)

    tf.dispose(t)

    return o 
}

// transform functions

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

function inversePermutation(permutation: Permute): Permute
{
    const inv = new Array<number>(permutation.length)
    for (let i = 0; i < permutation.length; i++) 
    {
        inv[permutation[i]] = i
    }

    return inv as Permute
}

function applyPermutation(newOffset: [number, number, number], permutation: Permute): [number, number, number] 
{
    const oldOffset: [number, number, number] = [0, 0, 0]

    oldOffset[permutation[0]] = newOffset[0]
    oldOffset[permutation[1]] = newOffset[1]
    oldOffset[permutation[2]] = newOffset[2]
    
    return oldOffset
}

// helper functions

function logTensor(name: string, tensor: tf.Tensor)
{
    console.log(name, tf.tidy(() => tensor.mean([0,1,2]).dataSync())) 
}

function logAnisotropicBidirectionalShadowMaps(shadowMaps: tf.Tensor3D)
{
    const unpack = new UnpackAnisotropicBidirectionalShadowMap(shadowMaps.shape)
    const size = shadowMaps.size

    console.log('shadowMap0', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[0]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMap1', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[1]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMap2', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[2]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMap3', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[3]]).mean([0,1,2]).dataSync())) 
}

function logExtendedAnisotropicBidirectionalShadowMaps(shadowMaps: tf.Tensor3D)
{
    const unpack = new UnpackExtendedAnisotropicBidirectionalShadowMap(shadowMaps.shape)
    const size = shadowMaps.size

    console.log('shadowMapX0', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[ 0]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMapX1', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[ 1]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMapX2', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[ 2]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMapX3', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[ 3]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMapY0', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[ 4]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMapY1', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[ 5]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMapY2', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[ 6]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMapY3', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[ 7]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMapZ0', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[ 8]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMapZ1', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[ 9]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMapZ2', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[10]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMapZ3', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[11]]).mean([0,1,2]).dataSync())) 
}

function runWebGLProgram(
    prog: GPGPUProgram, 
    inputs: tf.Tensor[], 
    dtype?: tf.DataType, 
    customValues?: number[][], 
    preventEagerUnpackingOfOutput?: boolean): tf.Tensor
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(prog, inputs, dtype, customValues, preventEagerUnpackingOfOutput)
    return tf.engine().makeTensorFromTensorInfo(info) 
}
