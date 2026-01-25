import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class GPGPUMinimaMap implements GPGPUProgram 
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

        ivec3 getCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec3 coords)
        {
            CellValues c;

            c.v000 = getA(coords - ivec3(1,1,1));
            c.v100 = getA(coords - ivec3(0,1,1));
            c.v010 = getA(coords - ivec3(1,0,1));
            c.v001 = getA(coords - ivec3(1,1,0));
            c.v011 = getA(coords - ivec3(1,0,0));
            c.v101 = getA(coords - ivec3(0,1,0));
            c.v110 = getA(coords - ivec3(0,0,1));
            c.v111 = getA(coords - ivec3(0,0,0));

            return c;
        }

        float getMinOnFaceX1(CellValues c)
        {
            return min4(c.v100, c.v110, c.v101, c.v111);
        }
    
        float getMinOnFaceY1(CellValues c)
        {
            return min4(c.v010, c.v011, c.v110, c.v111);
        }

        float getMinOnFaceZ1(CellValues c)
        {
            return min4(c.v001, c.v011, c.v101, c.v111);
        }

        void main()
        {
            ivec3 coords = getCoords();
            CellValues c = getValues(coords);

            float xMin = getMinOnFaceX1(c);
            float yMin = getMinOnFaceY1(c);
            float zMin = getMinOnFaceZ1(c);

            setOutput(vec4(xMin, yMin, zMin, 0.0));
        }
        `
    }
}

class GPGPUMinimaMap1 implements GPGPUProgram 
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

        ivec3 getCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec3 coords)
        {
            CellValues c;

            c.v000 = getA(coords - ivec3(1,1,1));
            c.v100 = getA(coords - ivec3(0,1,1));
            c.v010 = getA(coords - ivec3(1,0,1));
            c.v001 = getA(coords - ivec3(1,1,0));
            c.v011 = getA(coords - ivec3(1,0,0));
            c.v101 = getA(coords - ivec3(0,1,0));
            c.v110 = getA(coords - ivec3(0,0,1));
            c.v111 = getA(coords - ivec3(0,0,0));

            return c;
        }

        float getMinOnFaceX1(CellValues c000, CellValues c100)
        {
            float m0 = min4(c000.v100, c000.v110, c000.v101, c000.v111);
            float m1 = min4(c000.v100, c000.v110, c000.v101, c000.v011);
            float m2 = min4(c100.v001, c100.v010, c100.v100, c100.v011);

            m1 = min(m1, avg3(c000.v001, c000.v101, c000.v111));
            m1 = min(m1, avg3(c000.v001, c000.v011, c000.v111));
            m1 = min(m1, avg3(c000.v000, c000.v110, c000.v101));
            m1 = min(m1, avg3(c000.v010, c000.v001, c000.v111));
            m1 = min(m1, avg3(c000.v010, c000.v011, c000.v111));
            m1 = min(m1, avg3(c000.v010, c000.v110, c000.v111));

            m2 = min(m2, avg3(c100.v000, c100.v001, c100.v101));
            m2 = min(m2, avg3(c100.v000, c100.v100, c100.v101));
            m2 = min(m2, avg3(c100.v000, c100.v010, c100.v110));
            m2 = min(m2, avg3(c100.v000, c100.v100, c100.v110));
            m2 = min(m2, avg3(c100.v010, c100.v001, c100.v111));
            m2 = min(m2, avg3(c100.v000, c100.v110, c100.v101));

            return max3(m0, m1, m2);
        }
    
        float getMinOnFaceY1(CellValues c000, CellValues c010)
        {
            float m0 = min4(c000.v010, c000.v011, c000.v110, c000.v111);
            float m1 = min4(c000.v010, c000.v011, c000.v110, c000.v101);
            float m2 = min4(c010.v100, c010.v001, c010.v010, c010.v101);

            m1 = min(m1, avg3(c000.v100, c000.v110, c000.v111));
            m1 = min(m1, avg3(c000.v100, c000.v101, c000.v111));
            m1 = min(m1, avg3(c000.v000, c000.v011, c000.v110));
            m1 = min(m1, avg3(c000.v001, c000.v100, c000.v111));
            m1 = min(m1, avg3(c000.v001, c000.v101, c000.v111));
            m1 = min(m1, avg3(c000.v001, c000.v011, c000.v111));

            m2 = min(m2, avg3(c010.v000, c010.v100, c010.v110));
            m2 = min(m2, avg3(c010.v000, c010.v010, c010.v110));
            m2 = min(m2, avg3(c010.v000, c010.v001, c010.v011));
            m2 = min(m2, avg3(c010.v000, c010.v010, c010.v011));
            m2 = min(m2, avg3(c010.v001, c010.v100, c010.v111));
            m2 = min(m2, avg3(c010.v000, c010.v011, c010.v110));

            return max3(m0, m1, m2);
        }

        float getMinOnFaceZ1(CellValues c000, CellValues c001)
        {
            float m0 = min4(c000.v001, c000.v011, c000.v101, c000.v111);
            float m1 = min4(c000.v001, c000.v011, c000.v101, c000.v110);
            float m2 = min4(c001.v100, c001.v010, c001.v001, c001.v110);

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
            ivec3 coord = getCoords();

            CellValues c000 = getValues(coord + ivec3(0,0,0));
            CellValues c100 = getValues(coord + ivec3(1,0,0));
            CellValues c010 = getValues(coord + ivec3(0,1,0));
            CellValues c001 = getValues(coord + ivec3(0,0,1));

            float xMin = getMinOnFaceX1(c000, c100);
            float yMin = getMinOnFaceY1(c000, c010);
            float zMin = getMinOnFaceZ1(c000, c001);

            setOutput(vec4(xMin, yMin, zMin, 0.0));
        }
        `
    }
}

class GPGPUUpdateSlice implements GPGPUProgram 
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
        const [, outHeight, outWidth] = inputShape.map((x: number) => x + 1)
        this.outputShape = [outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec2 minCoords = ivec2(0);
        const ivec2 maxCoords = ivec2(${outWidth-1}, ${outHeight-1});

        ivec2 getCoords()
        {
            ivec4 coords = getOutputCoords();
            return ivec2(coords.y, coords.x);
        }

        vec4 getA(ivec2 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.y, coords.x, 0, 0);
        }

        vec4 getB(ivec2 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getB(coords.y, coords.x, 0, 0);
        }
                
        void main()
        {
            ivec2 coords = getCoords();

            vec4 c111 = getA(coords - ivec2(0,0));
            vec4 c011 = getA(coords - ivec2(1,0));
            vec4 c101 = getA(coords - ivec2(0,1));
            vec4 c001 = getA(coords - ivec2(1,1));

            vec4 c110 = getB(coords - ivec2(0,0));
            vec4 c010 = getB(coords - ivec2(1,0));
            vec4 c100 = getB(coords - ivec2(0,1));
            vec4 c000 = getB(coords - ivec2(1,1));

            float c011_x = max(c011.x, min(c010.z, max(c001.y, c000.z)));
            float c101_y = max(c101.y, min(c100.z, max(c001.x, c000.z)));

            c111.x = max(c111.x, min(c110.z, max(c101.y, c100.z)));
            c111.y = max(c111.y, min(c110.z, max(c011.x, c010.z)));
            c111.z = max(c111.z, min(c110.z, min(c011_x, c101_y)));

            setOutput(c111);
        }
        `
    }
}

class GPGPUUpdateMap implements GPGPUProgram 
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
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        float min3(float a, float b, float c) { return min(min(a, b), c); }

        ivec3 getCoords()
        {
            ivec5 coords = getOutputCoords();
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
                
        void main()
        {
            ivec3 coords = getCoords();

            vec4 c111 = getA(coords - ivec3(0,0,0));
            vec4 c011 = getB(coords - ivec3(1,0,0));
            vec4 c101 = getB(coords - ivec3(0,1,0));
            vec4 c110 = getB(coords - ivec3(0,0,1));
            
            c111.x = max(c111.x, min(c110.z, c011.x));
            c111.y = max(c111.y, min(c110.z, c101.y));
            c111.z = max(c111.z, min3(c110.z, c011.x, c101.y));

            setOutput(c111);
        }
        `
    }
}

class GPGPUMaximaMap implements GPGPUProgram 
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

        ivec3 getCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec3 coords)
        {
            CellValues c;

            c.v000 = getA(coords - ivec3(1,1,1));
            c.v100 = getA(coords - ivec3(0,1,1));
            c.v010 = getA(coords - ivec3(1,0,1));
            c.v001 = getA(coords - ivec3(1,1,0));
            c.v011 = getA(coords - ivec3(1,0,0));
            c.v101 = getA(coords - ivec3(0,1,0));
            c.v110 = getA(coords - ivec3(0,0,1));
            c.v111 = getA(coords - ivec3(0,0,0));

            return c;
        }

        float getMaxOnFaceX1(CellValues c)
        {
            return max4(c.v100, c.v110, c.v101, c.v111);
        }
    
        float getMaxOnFaceY1(CellValues c)
        {
            return max4(c.v010, c.v110, c.v011, c.v111);
        }

        float getMaxOnFaceZ1(CellValues c)
        {
            return max4(c.v001, c.v011, c.v101, c.v111);
        }

        void main()
        {
            ivec3 coords = getCoords();
            CellValues c = getValues(coords);

            float xMax = getMaxOnFaceX1(c);
            float yMax = getMaxOnFaceY1(c);
            float zMax = getMaxOnFaceZ1(c);

            setOutput(vec4(xMax, yMax, zMax, 1.0));
        }
        `
    }
}

class GPGPUMaximaMap1 implements GPGPUProgram 
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

        ivec3 getCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec3 coords)
        {
            CellValues c;

            c.v000 = getA(coords - ivec3(1,1,1));
            c.v100 = getA(coords - ivec3(0,1,1));
            c.v010 = getA(coords - ivec3(1,0,1));
            c.v001 = getA(coords - ivec3(1,1,0));
            c.v011 = getA(coords - ivec3(1,0,0));
            c.v101 = getA(coords - ivec3(0,1,0));
            c.v110 = getA(coords - ivec3(0,0,1));
            c.v111 = getA(coords - ivec3(0,0,0));

            return c;
        }

        float getMaxOnFaceX0(CellValues c)
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
    
        float getMaxOnFaceY0(CellValues c)
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

        float getMaxOnFaceZ0(CellValues c)
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

        float getMaxDiffOnFace0(CellValues c)
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
            ivec3 coords = getCoords();
            CellValues c = getValues(coords);

            float xMax = getMaxOnFaceX0(c);
            float yMax = getMaxOnFaceY0(c);
            float zMax = getMaxOnFaceZ0(c);
            float wMax = getMaxDiffOnFace0(c);

            setOutput(vec4(xMax, yMax, zMax, wMax));
        }
        `
    }
}

class GPGPUOcclusionMap implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = false

    constructor
    (
        inputShape: [number, number, number], 
    ) 
    {
        const [outDepth, outHeight, outWidth] = inputShape.map((x: number) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        ivec3 getCoords()
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
                
        void main()
        {
            ivec3 coords = getCoords();
            vec4 minValues = getA(coords);
            vec4 maxValues = getB(coords);

            bvec4 occlusion = greaterThanEqual(minValues, maxValues);
            occlusion.w = occlusion.w || all(occlusion.xyz);

            setOutput(float(occlusion.w));
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
    const minimaProgram = new GPGPUMinimaMap(volumeMap.shape)
    const maximaProgram = new GPGPUMaximaMap(volumeMap.shape)
    const updateProgram = new GPGPUUpdateMap(volumeMap.shape)
    const occlusionProgram = new GPGPUOcclusionMap(volumeMap.shape)

    let minimaMap = runProgram(minimaProgram, [volumeMap])
    const minimaMap0 = minimaMap.clone()

    console.log('minimaMap0', minimaMap.mean().dataSync())
    const numIterations = minimaMap.shape.reduce((a, x) => a + x, 0)

    for (let i = 0; i < numIterations; i++)
    {
        const temp = runProgram(updateProgram, [minimaMap0, minimaMap])
        tf.dispose(minimaMap)
        minimaMap = temp

        console.log(i)
        if (i % 1 === 0) await tf.nextFrame()
    }
    tf.dispose(minimaMap0)

    console.log('minimaMap', minimaMap.mean().dataSync())

    const maximaMap = runProgram(maximaProgram, [volumeMap])
    const occlusionMap = runProgram(occlusionProgram, [minimaMap, maximaMap])
    tf.dispose([minimaMap, maximaMap])

    console.log('occlusionMap', occlusionMap.mean().dataSync())

    return occlusionMap as tf.Tensor
}

export function computeOcclusionMap1(volumeMap: tf.Tensor3D) : tf.Tensor<tf.Rank>
{
    const minimaProgram = new GPGPUMinimaMap1(volumeMap.shape)
    const maximaProgram = new GPGPUMaximaMap1(volumeMap.shape)
    const updateProgram = new GPGPUUpdateSlice(volumeMap.shape)
    const occlusionProgram = new GPGPUOcclusionMap(volumeMap.shape)

    let minimaMap = runProgram(minimaProgram, [volumeMap])
    console.log('minimaMap0', minimaMap.mean().dataSync())
    let slices = tf.unstack(minimaMap, 0)
    minimaMap.dispose()

    for (let i = 1; i < slices.length; i++)
    {
        const updatedSlice = runProgram(updateProgram, [slices[i], slices[i-1]])
        tf.dispose(slices[i])
        slices[i] = updatedSlice
    }

    minimaMap = tf.stack(slices, 0)
    tf.dispose(slices)
    console.log('minimaMap', minimaMap.mean().dataSync())

    const maximaMap = runProgram(maximaProgram, [volumeMap])
    const occlusionMap = runProgram(occlusionProgram, [minimaMap, maximaMap])
    tf.dispose([minimaMap, maximaMap])

    console.log('occlusionMap', occlusionMap.mean().dataSync())

    return occlusionMap as tf.Tensor
}

export function computeOmniOcclusionMap(volumeMap: tf.Tensor3D) : tf.Tensor<tf.Rank>
{
    const map = tf.transpose(volumeMap, [2, 1, 0])

    const posOcclusion = computeOcclusionMap1(map)
    const negOcclusion = tf.tidy(() => tf.reverse(computeOcclusionMap1(tf.reverse(map))))
    tf.dispose(map)

    const occlusionMap = tf.maximum(posOcclusion, negOcclusion)
    tf.dispose([posOcclusion, negOcclusion])

    console.log('occlusionMap', occlusionMap.mean().dataSync())

    return occlusionMap as tf.Tensor
}