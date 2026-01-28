import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class GPGPUMinimaMaps implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor
    (
        volumeShape: [number, number, number], 
    ) 
    {
        const [inDepth, inHeight, inWidth] = volumeShape
        const [outDepth, outHeight, outWidth] = volumeShape.map((x: number) => x + 1)
        this.outputShape = [4, outDepth, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float avg3(float a, float b, float c) { return (1.0/3.0) * (a + b + c); }
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

        ivec4 getOutCoords()
        {
            ivec6 coords = getOutputCoords();
            return ivec4(coords.w, coords.z, coords.y, coords.x);
        }

        ivec3 getInCoords(ivec4 coords, int ox, int oy, int oz)
        {
            if (coords.w / 2 == 1) ox = 1-ox;
            if (coords.w % 2 == 1) oy = 1-oy;

            return coords.xyz - ivec3(ox, oy, oz);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec4 coords)
        {
            CellValues c;

            c.v000 = getA(getInCoords(coords, 1,1,1));
            c.v100 = getA(getInCoords(coords, 0,1,1));
            c.v010 = getA(getInCoords(coords, 1,0,1));
            c.v001 = getA(getInCoords(coords, 1,1,0));
            c.v011 = getA(getInCoords(coords, 1,0,0));
            c.v101 = getA(getInCoords(coords, 0,1,0));
            c.v110 = getA(getInCoords(coords, 0,0,1));
            c.v111 = getA(getInCoords(coords, 0,0,0));

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
            return min4(c.v001, c.v101, c.v011, c.v111);
        }

        void main()
        {
            ivec4 coords = getOutCoords();
            CellValues c = getValues(coords);

            float xMin = getMinOnFaceX(c);
            float yMin = getMinOnFaceY(c);
            float zMin = getMinOnFaceZ(c);

            setOutput(vec4(xMin, yMin, zMin, 0.0));
        }
        `
    }
}
class GPGPUMinimaMaps2 implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor
    (
        volumeShape: [number, number, number], 
    ) 
    {
        const [inDepth, inHeight, inWidth] = volumeShape
        const [outDepth, outHeight, outWidth] = volumeShape.map((x: number) => x + 1)
        this.outputShape = [4, outDepth, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float avg3(float a, float b, float c) { return (1.0/3.0) * (a + b + c); }
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

        ivec4 getOutCoords()
        {
            ivec6 coords = getOutputCoords();
            return ivec4(coords.w, coords.z, coords.y, coords.x);
        }

        ivec3 getInCoords(ivec4 coords, int ox, int oy, int oz)
        {
            if (coords.w / 2 == 1) ox = 1-ox;
            if (coords.w % 2 == 1) oy = 1-oy;

            return coords.xyz - ivec3(ox, oy, oz);
        }

        ivec4 getCellCoords(ivec4 coords, int ox, int oy, int oz)
        {
            if (coords.w / 2 == 1) ox = -ox;
            if (coords.w % 2 == 1) oy = -oy;
            
            return coords + ivec4(ox, oy, oz, 0);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec4 coords)
        {
            CellValues c;

            c.v000 =  getA(getInCoords(coords, 1,1,1));
            c.v100 =  getA(getInCoords(coords, 0,1,1));
            c.v010 =  getA(getInCoords(coords, 1,0,1));
            c.v001 =  getA(getInCoords(coords, 1,1,0));
            c.v011 =  getA(getInCoords(coords, 1,0,0));
            c.v101 =  getA(getInCoords(coords, 0,1,0));
            c.v110 =  getA(getInCoords(coords, 0,0,1));
            c.v111 =  getA(getInCoords(coords, 0,0,0));

            return c;
        }

        float getMinOnFaceX(CellValues c000, CellValues c100)
        {
            float t0 = min4(c000.v100, c000.v110, c000.v101, c000.v111);
            float t1 = min4(c000.v100, c000.v110, c000.v101, c000.v011);
            float t2 = min4(c100.v001, c100.v010, c100.v100, c100.v011);

            t1 = min(t1, avg3(c000.v001, c000.v101, c000.v111));
            t1 = min(t1, avg3(c000.v001, c000.v011, c000.v111));
            t1 = min(t1, avg3(c000.v000, c000.v101, c000.v110));
            t1 = min(t1, avg3(c000.v001, c000.v010, c000.v111));
            t1 = min(t1, avg3(c000.v010, c000.v110, c000.v111));
            t1 = min(t1, avg3(c000.v010, c000.v011, c000.v111));

            t2 = min(t2, avg3(c100.v000, c100.v010, c100.v110));
            t2 = min(t2, avg3(c100.v000, c100.v100, c100.v110));
            t2 = min(t2, avg3(c100.v000, c100.v001, c100.v101));
            t2 = min(t2, avg3(c100.v000, c100.v100, c100.v101));
            t2 = min(t2, avg3(c100.v001, c100.v010, c100.v111));
            t2 = min(t2, avg3(c100.v000, c100.v101, c100.v110));

            return max3(t0, t1, t2);
        }

        float getMinOnFaceY(CellValues c000, CellValues c010)
        {
            float t0 = min4(c000.v010, c000.v110, c000.v011, c000.v111);
            float t1 = min4(c000.v010, c000.v110, c000.v011, c000.v101);
            float t2 = min4(c010.v001, c010.v100, c010.v010, c010.v101);

            t1 = min(t1, avg3(c000.v001, c000.v011, c000.v111));
            t1 = min(t1, avg3(c000.v001, c000.v101, c000.v111));
            t1 = min(t1, avg3(c000.v000, c000.v011, c000.v110));
            t1 = min(t1, avg3(c000.v001, c000.v100, c000.v111));
            t1 = min(t1, avg3(c000.v100, c000.v110, c000.v111));
            t1 = min(t1, avg3(c000.v100, c000.v101, c000.v111));

            t2 = min(t2, avg3(c010.v000, c010.v100, c010.v110));
            t2 = min(t2, avg3(c010.v000, c010.v010, c010.v110));
            t2 = min(t2, avg3(c010.v000, c010.v001, c010.v011));
            t2 = min(t2, avg3(c010.v000, c010.v010, c010.v011));
            t2 = min(t2, avg3(c010.v001, c010.v100, c010.v111));
            t2 = min(t2, avg3(c010.v000, c010.v011, c010.v110));

            return max3(t0, t1, t2);
        }
            
        float getMinOnFaceZ(CellValues c000, CellValues c001)
        {
            float t0 = min4(c000.v001, c000.v011, c000.v101, c000.v111);
            float t1 = min4(c000.v001, c000.v011, c000.v101, c000.v110);
            float t2 = min4(c001.v100, c001.v010, c001.v001, c001.v110);

            t1 = min(t1, avg3(c000.v100, c000.v101, c000.v111));
            t1 = min(t1, avg3(c000.v100, c000.v110, c000.v111));
            t1 = min(t1, avg3(c000.v000, c000.v011, c000.v101));
            t1 = min(t1, avg3(c000.v010, c000.v100, c000.v111));
            t1 = min(t1, avg3(c000.v010, c000.v110, c000.v111));
            t1 = min(t1, avg3(c000.v010, c000.v011, c000.v111));

            t2 = min(t2, avg3(c001.v000, c001.v100, c001.v101));
            t2 = min(t2, avg3(c001.v000, c001.v001, c001.v101));
            t2 = min(t2, avg3(c001.v000, c001.v010, c001.v011));
            t2 = min(t2, avg3(c001.v000, c001.v001, c001.v011));
            t2 = min(t2, avg3(c001.v010, c001.v100, c001.v111));
            t2 = min(t2, avg3(c001.v000, c001.v011, c001.v101));

            return max3(t0, t1, t2);
        }

        void main()
        {
            ivec4 outCoords = getOutCoords();

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

class GPGPUMaximaMaps implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor
    (
        volumeShape: [number, number, number], 
    ) 
    {
        const [inDepth, inHeight, inWidth] = volumeShape
        const [outDepth, outHeight, outWidth] = volumeShape.map((x: number) => x + 1)
        this.outputShape = [4, outDepth, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float avg3(float a, float b, float c) { return (1.0/3.0) * (a + b + c); }

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

        ivec4 getOutCoords()
        {
            ivec6 coords = getOutputCoords();
            return ivec4(coords.w, coords.z, coords.y, coords.x);
        }

        ivec3 getInCoords(ivec4 coords, int ox, int oy, int oz)
        {
            if (coords.w / 2 == 1) ox = 1-ox;
            if (coords.w % 2 == 1) oy = 1-oy;

            return coords.xyz - ivec3(ox, oy, oz);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec4 coords)
        {
            CellValues c;

            c.v000 = getA(getInCoords(coords, 1,1,1));
            c.v100 = getA(getInCoords(coords, 0,1,1));
            c.v010 = getA(getInCoords(coords, 1,0,1));
            c.v001 = getA(getInCoords(coords, 1,1,0));
            c.v011 = getA(getInCoords(coords, 1,0,0));
            c.v101 = getA(getInCoords(coords, 0,1,0));
            c.v110 = getA(getInCoords(coords, 0,0,1));
            c.v111 = getA(getInCoords(coords, 0,0,0));

            return c;
        }

        float getMaxOnFaceX(CellValues c)
        {
            float m = -1.0/0.0;

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
            float m = -1.0/0.0;

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
            float m = -1.0/0.0;

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
            float m = -1.0/0.0;

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
            ivec4 coords = getOutCoords();
            CellValues c = getValues(coords);

            float xMax = getMaxOnFaceX(c);
            float yMax = getMaxOnFaceY(c);
            float zMax = getMaxOnFaceZ(c);
            float wMax = getMaxOnCell(c);

            setOutput(vec4(xMax, yMax, zMax, wMax));
        }
        `
    }
}

class GPGPUUpdateSlices implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor
    (
        volumeShape: [number, number, number], 
    ) 
    {
        const [, outHeight, outWidth] = volumeShape.map((x: number) => x + 1)
        this.outputShape = [4, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, 3);

        float min3(float a, float b, float c) { return min(min(a, b), c); }

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        ivec3 getInCoords(ivec3 coords, int ox, int oy)
        {
            if (coords.z / 2 == 1) ox = -ox;
            if (coords.z % 2 == 1) oy = -oy;
            
            return coords - ivec3(ox, oy, 0);
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
            ivec3 coords = getOutCoords();

            vec4 c111 = getA(getInCoords(coords, 0,0));
            vec4 c011 = getA(getInCoords(coords, 1,0));
            vec4 c101 = getA(getInCoords(coords, 0,1));
            vec4 c001 = getA(getInCoords(coords, 1,1));

            vec4 c110 = getB(getInCoords(coords, 0,0));
            vec4 c010 = getB(getInCoords(coords, 1,0));
            vec4 c100 = getB(getInCoords(coords, 0,1));
            vec4 c000 = getB(getInCoords(coords, 1,1));

            c111.x = getMinOnFaceX(c111, c110, c101, c100);
            c111.y = getMinOnFaceY(c111, c110, c011, c010);
            c111.z = getMinOnFaceZ(c111, c110, c101, c011, c100, c010, c001, c000);

            setOutput(c111);
        }
        `
    }
}

class GPGPUOcclusionMaps implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor
    (
        volumeShape: [number, number, number], 
    ) 
    {
        const [outDepth, outHeight, outWidth] = volumeShape.map((x: number) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getA(ivec3 coords, int i)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(i, coords.z, coords.y, coords.x, 0, 0);
        }

        vec4 getB(ivec3 coords, int i)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getB(i, coords.z, coords.y, coords.x, 0, 0);
        }

        int getOcclusion(ivec3 coords, int i)
        {
            vec4 minValues = getA(coords, i);
            vec4 maxValues = getB(coords, i);

            bvec4 tests = greaterThanEqual(minValues, maxValues);
            bool occlusion = all(tests.xyz) || tests.w;

            return int(occlusion);
        }
                
        void main()
        {
            ivec3 coords = getOutCoords();
            
            setOutput(vec4(
                getOcclusion(coords, 0),
                getOcclusion(coords, 1),
                getOcclusion(coords, 2),
                getOcclusion(coords, 3)
            ));
        }
        `
    }
}
class GPGPUReverse5d implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor
    (
        volumeShape: [number, number, number], 
    ) 
    {
        const [outDepth, outHeight, outWidth] = volumeShape.map((x: number) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x, 0, 0);
        }
                
        void main()
        {
            ivec3 coords = getOutCoords();
            ivec3 revCoords = maxCoords - coords;
    
            setOutput(getA(revCoords));    
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

export function computeOneWayOcclusionMaps(volumeMap: tf.Tensor3D) : tf.Tensor5D
{
    const minimaProgram = new GPGPUMinimaMaps2(volumeMap.shape)
    const minimaStart = runProgram(minimaProgram, [volumeMap])
    console.log('minimaStart', tf.tidy(() => minimaStart.unstack(0)[0].mean([0,1,2]).dataSync()))

    const updateProgram = new GPGPUUpdateSlices(volumeMap.shape)
    const slices = tf.unstack(minimaStart, 1)
    minimaStart.dispose()

    for (let i = 1; i < slices.length; i++)
    {
        const updatedSlice = runProgram(updateProgram, [slices[i], slices[i-1]])
        tf.dispose(slices[i])
        slices[i] = updatedSlice
    }

    const minima = tf.stack(slices, 1); 
    tf.dispose(slices)
    console.log('minima', tf.tidy(() => minima.unstack(0)[0].mean([0,1,2]).dataSync()))

    const maximaProgram = new GPGPUMaximaMaps(volumeMap.shape)
    const maxima = runProgram(maximaProgram, [volumeMap])
    console.log('maxima', tf.tidy(() => maxima.unstack(0)[0].mean([0,1,2]).dataSync()))

    const occlusionProgram = new GPGPUOcclusionMaps(volumeMap.shape)
    const occlusion = runProgram(occlusionProgram, [minima, maxima])
    tf.dispose([minima, maxima])

    console.log('occlusionMap', tf.tidy(() => occlusion.mean([0,1,2]).dataSync()))

    return occlusion as tf.Tensor5D
}

export function computeOcclusionMaps(volumeMap: tf.Tensor3D): tf.Tensor3D
{
    const occlusionMapPos = tf.tidy(() => computeOneWayOcclusionMaps(volumeMap))

    const reverse5d = new GPGPUReverse5d(volumeMap.shape)
    const occlusionMapNeg = tf.tidy(() => runProgram(reverse5d, [computeOneWayOcclusionMaps(tf.reverse(volumeMap))]))

    const occlusionMap = tf.maximum(occlusionMapPos, occlusionMapNeg)
    tf.dispose([occlusionMapPos, occlusionMapNeg])

    console.log('occlusionMap',  tf.tidy(() => occlusionMap.mean([0,1,2]).dataSync()))

    return occlusionMap as tf.Tensor3D
}
