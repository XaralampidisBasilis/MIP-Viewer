import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class GPGPUMinimaMaps0 implements GPGPUProgram 
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

        float avg3(float a, float b, float c) { return (1.0 / 3.0) * (a + b + c); }
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

            return coords.xyz + ivec3(ox, oy, oz);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec4 coords)
        {
            CellValues c;

            coords = coords - 1;

            c.v000 = getA(getInCoords(coords, 0,0,0));
            c.v100 = getA(getInCoords(coords, 1,0,0));
            c.v010 = getA(getInCoords(coords, 0,1,0));
            c.v001 = getA(getInCoords(coords, 0,0,1));
            c.v011 = getA(getInCoords(coords, 0,1,1));
            c.v101 = getA(getInCoords(coords, 1,0,1));
            c.v110 = getA(getInCoords(coords, 1,1,0));
            c.v111 = getA(getInCoords(coords, 1,1,1));

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

class GPGPUMaximaMaps0 implements GPGPUProgram 
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

            return ivec3(coords.x + ox, coords.y + oy, coords.z + oz);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec4 coords)
        {
            CellValues c;

            coords = coords - 1;

            c.v000 = getA(getInCoords(coords, 0,0,0));
            c.v100 = getA(getInCoords(coords, 1,0,0));
            c.v010 = getA(getInCoords(coords, 0,1,0));
            c.v001 = getA(getInCoords(coords, 0,0,1));
            c.v011 = getA(getInCoords(coords, 0,1,1));
            c.v101 = getA(getInCoords(coords, 1,0,1));
            c.v110 = getA(getInCoords(coords, 1,1,0));
            c.v111 = getA(getInCoords(coords, 1,1,1));

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
            ivec4 coords = getOutCoords();
            CellValues c = getValues(coords);

            float xMax = getMaxOnFaceX(c);
            float yMax = getMaxOnFaceY(c);
            float zMax = getMaxOnFaceZ(c);
            float wMax = getMaxOnCell(c);

            setOutput(vec4(xMax, yMax, zMax, 1.0));
        }
        `
    }
}

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

            return coords.xyz + ivec3(ox, oy, oz);
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

            coords = coords - 1;

            c.v000 =  getA(getInCoords(coords, 0,0,0));
            c.v100 =  getA(getInCoords(coords, 1,0,0));
            c.v010 =  getA(getInCoords(coords, 0,1,0));
            c.v001 =  getA(getInCoords(coords, 0,0,1));
            c.v011 =  getA(getInCoords(coords, 0,1,1));
            c.v101 =  getA(getInCoords(coords, 1,0,1));
            c.v110 =  getA(getInCoords(coords, 1,1,0));
            c.v111 =  getA(getInCoords(coords, 1,1,1));

            return c;
        }

        float getMinOnFaceX(CellValues c000, CellValues c100)
        {
            float m0 = min4(c000.v100, c000.v110, c000.v101, c000.v111);
            float m1 = min4(c000.v100, c000.v110, c000.v101, c000.v011);
            float m2 = min4(c100.v001, c100.v010, c100.v100, c100.v011);

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
            float m0 = min4(c000.v010, c000.v110, c000.v011, c000.v111);
            float m1 = min4(c000.v010, c000.v110, c000.v011, c000.v101);
            float m2 = min4(c010.v001, c010.v100, c010.v010, c010.v101);

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

            return coords.xyz + ivec3(ox, oy, oz);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec4 coords)
        {
            CellValues c;

            coords = coords - 1;

            c.v000 = getA(getInCoords(coords, 0,0,0));
            c.v100 = getA(getInCoords(coords, 1,0,0));
            c.v010 = getA(getInCoords(coords, 0,1,0));
            c.v001 = getA(getInCoords(coords, 0,0,1));
            c.v011 = getA(getInCoords(coords, 0,1,1));
            c.v101 = getA(getInCoords(coords, 1,0,1));
            c.v110 = getA(getInCoords(coords, 1,1,0));
            c.v111 = getA(getInCoords(coords, 1,1,1));

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

class GPGPUUpdateMinimaSlices implements GPGPUProgram 
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
            
            return coords + ivec3(ox, oy, 0);
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
            float m10 = c100.z;

            m10 = max(c101.y, m10);
            m10 = min(c110.z, m10);
            m10 = max(c111.x, m10);

            return m10;
        }

        float getMinOnFaceY(vec4 c111, vec4 c110, vec4 c011, vec4 c010)
        {
            float m01 = c010.z;

            m01 = max(c011.x, m01);
            m01 = min(c110.z, m01);
            m01 = max(c111.y, m01);

            return m01;
        }

        float getMinOnFaceZ(vec4 c111, vec4 c110, vec4 c101, vec4 c011, vec4 c100, vec4 c010, vec4 c001, vec4 c000)
        {
            float m00, m01, m10, m11;

            m00 = c000.z;

            m01 = max(c001.y, m00);
            m01 = min(c010.z, m01);
            m01 = max(c011.x, m01);

            m10 = max(c001.x, m00);
            m10 = min(c100.z, m10);
            m10 = max(c101.y, m10);

            m11 = min(m01, m10);
            m11 = min(c110.z, m11);
            m11 = max(c111.z, m11);

            return m11;
        }
                
        void main()
        {
            ivec3 coords = getOutCoords();

            vec4 c111 = getA(getInCoords(coords, -0,-0));
            vec4 c011 = getA(getInCoords(coords, -1,-0));
            vec4 c101 = getA(getInCoords(coords, -0,-1));
            vec4 c001 = getA(getInCoords(coords, -1,-1));
            vec4 c110 = getB(getInCoords(coords, -0,-0));
            vec4 c010 = getB(getInCoords(coords, -1,-0));
            vec4 c100 = getB(getInCoords(coords, -0,-1));
            vec4 c000 = getB(getInCoords(coords, -1,-1));

            c111.x = getMinOnFaceX(c111, c110, c101, c100);
            c111.y = getMinOnFaceY(c111, c110, c011, c010);
            c111.z = getMinOnFaceZ(c111, c110, c101, c011, c100, c010, c001, c000);

            setOutput(c111);
        }
        `
    }
}

class GPGPUUpdateMinimaMaps implements GPGPUProgram 
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
        const [outDepth, outHeight, outWidth] = inputShape.map((x: number) => x + 1)
        this.outputShape = [4, outDepth, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec4 minCoords = ivec4(0);
        const ivec4 maxCoords = ivec4(${outWidth-1}, ${outHeight-1}, ${outDepth-1}, 3);

        float min3(float a, float b, float c) { return min(min(a, b), c); }

        ivec4 getOutCoords()
        {
            ivec6 coords = getOutputCoords();
            return ivec4(coords.w, coords.z, coords.y, coords.x);
        }

        ivec4 getInCoords(ivec4 coords, int ox, int oy, int oz)
        {
            if (coords.w / 2 == 1) ox = -ox;
            if (coords.w % 2 == 1) oy = -oy;
            
            return coords + ivec4(ox, oy, oz, 0);
        }

        vec4 getA(ivec4 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.w, coords.z, coords.y, coords.x, 0, 0);
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
            ivec4 coords = getOutCoords();

            vec4 c111 = getA(getInCoords(coords, -0,-0,-0));
            vec4 c011 = getA(getInCoords(coords, -1,-0,-0));
            vec4 c101 = getA(getInCoords(coords, -0,-1,-0));
            vec4 c001 = getA(getInCoords(coords, -1,-1,-0));
            vec4 c110 = getA(getInCoords(coords, -0,-0,-1));
            vec4 c010 = getA(getInCoords(coords, -1,-0,-1));
            vec4 c100 = getA(getInCoords(coords, -0,-1,-1));
            vec4 c000 = getA(getInCoords(coords, -1,-1,-1));

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
    packedOutput = false

    constructor
    (
        volumeShape: [number, number, number], 
    ) 
    {
        const [outDepth, outHeight, outWidth] = volumeShape.map((x: number) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});
                    
        ivec3 getOutCoords()
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getA(ivec3 coords, int wCoord)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(wCoord, coords.z, coords.y, coords.x, 0, 0);
        }

        vec4 getB(ivec3 coords, int wCoord)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getB(wCoord, coords.z, coords.y, coords.x, 0, 0);
        }

        uint getOcclusion(ivec3 coords, int wCoord)
        {
            vec4 minValues = getA(coords, wCoord);
            vec4 maxValues = getB(coords, wCoord);

            bvec4 tests = greaterThanEqual(minValues, maxValues);
            bool occlusion = all(tests.xyz) || tests.w;

            return uint(occlusion);
        }

        uint pack(uint o0, uint o1, uint o2, uint o3) 
        { 
            return (o0 << 0) | (o1 << 1) | (o2 << 2) | (o3 << 3);
        }
 
        void main()
        {
            ivec3 coords = getOutCoords();

            uint o0 = getOcclusion(coords, 0);
            uint o1 = getOcclusion(coords, 1);
            uint o2 = getOcclusion(coords, 2);
            uint o3 = getOcclusion(coords, 3);

            uint o = pack(o0, o1, o2, o3);

            setOutput(float(o));
        }
        `
    }
}

class GPGPUUniteOcclusionMaps implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor
    (
        volumeShape: [number, number, number], 
    ) 
    {
        const [outDepth, outHeight, outWidth] = volumeShape.map((x: number) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});
        
        uint toUintRounded(float f) { return uint(floor(f + 0.5)); }

        ivec3 getOutCoords()
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        float getB(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getB(coords.z, coords.y, coords.x);
        }
 
        void main()
        {
            ivec3 coords = getOutCoords();

            uint oA = toUintRounded(clamp(getA(coords), 0.0, 15.0));
            uint oB = toUintRounded(clamp(getB(coords), 0.0, 15.0));
            uint o = oA | oB;

            setOutput(float(o));
        }
        `
    }
}

class GPGPUPackOcclusionMaps implements GPGPUProgram 
{
    variableNames = ['A', 'B', 'C']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor
    (
        volumeShape: [number, number, number], 
    ) 
    {
        const [outDepth, outHeight, outWidth] = volumeShape.map((x: number) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});
           
        uint toUintRounded(float f) { return uint(floor(f + 0.5)); }

        ivec3 getOutCoords()
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        float getB(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getB(coords.z, coords.y, coords.x);
        }

        float getC(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getC(coords.z, coords.y, coords.x);
        }

        int pack(uint oA, uint oB, uint oC)
        {
            uint p = (oA << 0) | (oB << 4) | (oC << 8); // 0..4095
            return int(p) - 2048; // -2048..2047 in half float precision 
        }
 
        void main()
        {
            ivec3 coords = getOutCoords();

            uint oA = toUintRounded(clamp(getA(coords), 0.0, 15.0));
            uint oB = toUintRounded(clamp(getB(coords), 0.0, 15.0));
            uint oC = toUintRounded(clamp(getC(coords), 0.0, 15.0));

            setOutput(float(pack(oA, oB, oC)));
        }
        `
    }
}

class GPGPUUnpackOccupancyMap0 implements GPGPUProgram 
{
    variableNames = ['A'] 
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false
    customUniforms = [{ name: 'uMap', type: 'int' as const }]

    constructor(volumeShape: [number, number, number]) 
    {
        const [outDepth, outHeight, outWidth] = volumeShape.map(x => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        int toIntRounded(float f) { return int(f >= 0.0 ? floor(f + 0.5) : ceil(f - 0.5)); }
        uint toUintRounded(float f) { return uint(floor(f + 0.5)); }

        ivec3 getOutCoords() 
        {
            ivec3 c = getOutputCoords();
            return ivec3(c.z, c.y, c.x);
        }

        float getA(ivec3 coords) 
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        void main() 
        {
            ivec3 coords = getOutCoords();

            uint u = toUintRounded(clamp(getA(coords), 0.0, 15.0));
            uint o = (u >> uMap) & 1u;

            setOutput(float(o));
        }
        `
    }
}

class GPGPUUnpackOccupancyMap implements GPGPUProgram 
{
    variableNames = ['A'] 
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false
    customUniforms = [{ name: 'uMap', type: 'int' as const }]

    constructor(volumeShape: [number, number, number]) 
    {
        const [outDepth, outHeight, outWidth] = volumeShape.map(x => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        int toIntRounded(float f) { return int(f >= 0.0 ? floor(f + 0.5) : ceil(f - 0.5)); }

        ivec3 getOutCoords() 
        {
            ivec3 c = getOutputCoords();
            return ivec3(c.z, c.y, c.x);
        }

        float getA(ivec3 coords) 
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        uint unpack(int p) 
        {
            int i = clamp(p, -2048, 2047); // -2048..2047 
            return uint(i + 2048); // 0..4095
        }

        void main() 
        {
            ivec3 coords = getOutCoords();

            int p = toIntRounded(clamp(getA(coords), -2048.0, 2047.0));
            uint u = unpack(p);
            uint o = (u >> uMap) & 1u;

            setOutput(float(o));
        }
        `
    }
}

class GPGPUReverse implements GPGPUProgram 
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

function runProgram(prog: GPGPUProgram, inputs: tf.Tensor[], outputDtype?: tf.DataType, customUniformValues?: number[][], preventEagerUnpackingOfOutput?: boolean): tf.Tensor
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(prog, inputs, outputDtype, customUniformValues, preventEagerUnpackingOfOutput)
    return tf.engine().makeTensorFromTensorInfo(info) as tf.Tensor
}

export function computeOneWayOcclusionMaps0(volumeMap: tf.Tensor3D) : tf.Tensor5D
{
    const minimaProgram = new GPGPUMinimaMaps0(volumeMap.shape)
    const minimaStart = runProgram(minimaProgram, [volumeMap], 'float32', [], true)
    // console.log('minimaStart', tf.tidy(() => minimaStart.unstack(0)[0].mean([0,1,2]).dataSync()))

    const updateProgram = new GPGPUUpdateMinimaSlices(volumeMap.shape)
    const minimaSlices = tf.unstack(minimaStart, 1)
    minimaStart.dispose()

    for (let i = 1; i < minimaSlices.length; i++)
    {
        const updatedSlice = runProgram(updateProgram, [minimaSlices[i], minimaSlices[i-1]], 'float32', [], true)
        tf.dispose(minimaSlices[i])
        minimaSlices[i] = updatedSlice
    }

    const minimaMap = tf.stack(minimaSlices, 1); 
    tf.dispose(minimaSlices)
    // console.log('minimaMap', tf.tidy(() => minimaMap.unstack(0)[0].mean([0,1,2]).dataSync()))

    const maximaProgram = new GPGPUMaximaMaps0(volumeMap.shape)
    const maximaMap = runProgram(maximaProgram, [volumeMap], 'float32', [], true)
    // console.log('maximaMap', tf.tidy(() => maximaMap.unstack(0)[0].mean([0,1,2]).dataSync()))

    const occlusionProgram = new GPGPUOcclusionMaps(volumeMap.shape)
    const occlusionMap = runProgram(occlusionProgram, [minimaMap, maximaMap], 'int32', [], false)
    tf.dispose([minimaMap, maximaMap])
    // console.log('occlusionMap', tf.tidy(() => occlusionMap.mean([0,1,2]).dataSync()))

    return occlusionMap as tf.Tensor5D
}

export function computeOneWayOcclusionMaps(volumeMap: tf.Tensor3D) : tf.Tensor5D
{
    const minimaProgram = new GPGPUMinimaMaps(volumeMap.shape)
    const minimaStart = runProgram(minimaProgram, [volumeMap], 'float32', [], true)
    // console.log('minimaStart', tf.tidy(() => minimaStart.unstack(0)[0].mean([0,1,2]).dataSync()))

    const updateProgram = new GPGPUUpdateMinimaSlices(volumeMap.shape)
    const minimaSlices = tf.unstack(minimaStart, 1)
    minimaStart.dispose()

    for (let i = 1; i < minimaSlices.length; i++)
    {
        const updatedSlice = runProgram(updateProgram, [minimaSlices[i], minimaSlices[i-1]], 'float32', [], true)
        tf.dispose(minimaSlices[i])
        minimaSlices[i] = updatedSlice
    }

    const minimaMap = tf.stack(minimaSlices, 1); 
    tf.dispose(minimaSlices)
    // console.log('minimaMap', tf.tidy(() => minimaMap.unstack(0)[0].mean([0,1,2]).dataSync()))

    const maximaProgram = new GPGPUMaximaMaps(volumeMap.shape)
    const maximaMap = runProgram(maximaProgram, [volumeMap], 'float32', [], true)
    // console.log('maximaMap', tf.tidy(() => maximaMap.unstack(0)[0].mean([0,1,2]).dataSync()))

    const occlusionProgram = new GPGPUOcclusionMaps(volumeMap.shape)
    const occlusionMap = runProgram(occlusionProgram, [minimaMap, maximaMap], 'int32', [], false)
    tf.dispose([minimaMap, maximaMap])
    // console.log('occlusionMap', tf.tidy(() => occlusionMap.mean([0,1,2]).dataSync()))

    return occlusionMap as tf.Tensor5D
}

export async function computeOneWayOcclusionMapsAsync(volumeMap: tf.Tensor3D) : Promise<tf.Tensor5D>
{
    const updateProgram = new GPGPUUpdateMinimaMaps(volumeMap.shape)
    const minimaProgram = new GPGPUMinimaMaps(volumeMap.shape)
    let minimaMap = runProgram(minimaProgram, [volumeMap], 'float32', [], true)
    // console.log('minimaMap', tf.tidy(() => minimaMap.unstack(0)[0].mean([0,1,2]).dataSync()))

    for (let i = 0; i <= volumeMap.shape[0]; i++)
    {
        const temp = runProgram(updateProgram, [minimaMap], 'float32', [], true)
        tf.dispose(minimaMap)
        minimaMap = temp

        await tf.nextFrame()
    }
    // console.log('minimaMap', tf.tidy(() => minimaMap.unstack(0)[0].mean([0,1,2]).dataSync()))

    const maximaProgram = new GPGPUMaximaMaps(volumeMap.shape)
    const maximaMap = runProgram(maximaProgram, [volumeMap], 'float32', [], true)
    // console.log('maximaMap', tf.tidy(() => maximaMap.unstack(0)[0].mean([0,1,2]).dataSync()))

    const occlusionProgram = new GPGPUOcclusionMaps(volumeMap.shape)
    const occlusionMap = runProgram(occlusionProgram, [minimaMap, maximaMap], 'int32', [], false)
    tf.dispose([minimaMap, maximaMap])
    // console.log('occlusionMap', tf.tidy(() => occlusionMap.mean([0,1,2]).dataSync()))

    return occlusionMap as tf.Tensor5D
}

export function computeAnisotropicOcclusionMaps0(volumeMap: tf.Tensor3D): tf.Tensor3D
{
    const occlusionMapA = tf.tidy(() => tf.reverse(computeOneWayOcclusionMaps0(tf.reverse(volumeMap))))
    const occlusionMapB = tf.tidy(() => computeOneWayOcclusionMaps0(volumeMap))

    const logicalOr = new GPGPUUniteOcclusionMaps(volumeMap.shape)
    const occlusionMap = runProgram(logicalOr, [occlusionMapA, occlusionMapB], 'int32', [], false)
    tf.dispose([occlusionMapA, occlusionMapB])
    
    // console.log('occlusionMap0', tf.tidy(() => occlusionMap.floorDiv(1 << 0).mod(2).mean([0,1,2]).dataSync()))
    // console.log('occlusionMap1', tf.tidy(() => occlusionMap.floorDiv(1 << 1).mod(2).mean([0,1,2]).dataSync()))
    // console.log('occlusionMap2', tf.tidy(() => occlusionMap.floorDiv(1 << 2).mod(2).mean([0,1,2]).dataSync()))
    // console.log('occlusionMap3', tf.tidy(() => occlusionMap.floorDiv(1 << 3).mod(2).mean([0,1,2]).dataSync()))

    return occlusionMap as tf.Tensor3D
}

export function computeAnisotropicOcclusionMaps(volumeMap: tf.Tensor3D): tf.Tensor3D
{
    const occlusionMapA = tf.tidy(() => tf.reverse(computeOneWayOcclusionMaps(tf.reverse(volumeMap))))
    const occlusionMapB = tf.tidy(() => computeOneWayOcclusionMaps(volumeMap))

    const logicalOr = new GPGPUUniteOcclusionMaps(volumeMap.shape)
    const occlusionMap = runProgram(logicalOr, [occlusionMapA, occlusionMapB], 'float32', [], false)
    tf.dispose([occlusionMapA, occlusionMapB])

    const unpack = new GPGPUUnpackOccupancyMap0(volumeMap.shape)
    console.log('occlusionMap0', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[0]]).mean([0,1,2]).dataSync()))
    console.log('occlusionMap1', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[1]]).mean([0,1,2]).dataSync()))
    console.log('occlusionMap2', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[2]]).mean([0,1,2]).dataSync()))
    console.log('occlusionMap3', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[3]]).mean([0,1,2]).dataSync()))

    return occlusionMap as tf.Tensor3D
}

export function computeExtendedAnisotropicOcclusionMaps(volumeMap: tf.Tensor3D): tf.Tensor3D
{
    const occlusionMapX = tf.tidy(() => tf.transpose(computeAnisotropicOcclusionMaps(tf.transpose(volumeMap, [2,1,0])), [2,1,0]))
    const occlusionMapY = tf.tidy(() => tf.transpose(computeAnisotropicOcclusionMaps(tf.transpose(volumeMap, [1,0,2])), [1,0,2]))
    const occlusionMapZ = tf.tidy(() => computeAnisotropicOcclusionMaps(volumeMap))

    const pack = new GPGPUPackOcclusionMaps(volumeMap.shape)
    const occlusionMap = runProgram(pack, [occlusionMapX, occlusionMapY, occlusionMapZ], 'int32', [], false)
    tf.dispose([occlusionMapX, occlusionMapY, occlusionMapZ])

    const unpack = new GPGPUUnpackOccupancyMap(volumeMap.shape)
    console.log('occlusionMapX0', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[ 0]]).mean([0,1,2]).dataSync()))
    console.log('occlusionMapX1', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[ 1]]).mean([0,1,2]).dataSync()))
    console.log('occlusionMapX2', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[ 2]]).mean([0,1,2]).dataSync()))
    console.log('occlusionMapX3', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[ 3]]).mean([0,1,2]).dataSync()))
    console.log('occlusionMapY0', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[ 4]]).mean([0,1,2]).dataSync()))
    console.log('occlusionMapY1', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[ 5]]).mean([0,1,2]).dataSync()))
    console.log('occlusionMapY2', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[ 6]]).mean([0,1,2]).dataSync()))
    console.log('occlusionMapY3', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[ 7]]).mean([0,1,2]).dataSync()))
    console.log('occlusionMapZ0', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[ 8]]).mean([0,1,2]).dataSync()))
    console.log('occlusionMapZ1', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[ 9]]).mean([0,1,2]).dataSync()))
    console.log('occlusionMapZ2', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[10]]).mean([0,1,2]).dataSync()))
    console.log('occlusionMapZ3', tf.tidy(() => runProgram(unpack, [occlusionMap], 'float32', [[11]]).mean([0,1,2]).dataSync()))
    
    
    // 0.64208984375, 0.64208984375, 
    // 0.65673828125, 0.65673828125, 
    // 0.66748046875, 0.6669921875,  
    // 0.63134765625, 0.630859375,   
    // 0.83251953125, 0.83251953125, 
    // 0.82763671875, 0.82763671875, 
    // 0.81982421875, 0.81982421875, 
    // 0.77197265625, 0.77197265625, 
    // 0.724609375,   0.724609375,   
    // 0.7119140625,  0.7119140625,  
    // 0.73486328125, 0.73486328125, 
    // 0.71337890625, 0.71337890625, 

    
    return occlusionMap as tf.Tensor3D
}