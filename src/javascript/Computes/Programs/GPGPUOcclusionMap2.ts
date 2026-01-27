import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

/**
 * Rewrite v variables (e.g. v011) by permuting the 3 digits.
 * Digits are assumed to be in (x,y,z) order: v[x][y][z].
 *
 * @param {string} code
 * @param {[number, number, number]} perm  e.g. [1,0,2] means (x,y,z)->(y,x,z)
 * @returns {string}
 */
function transformCode(code: string, perm: [number, number, number]) : string
{
    code.replace(/\b([vc])([01])([01])([01])\b/g, (_, prefix, x, y, z) => 
    {
        const d = [x, y, z];
        return prefix + d[perm[0]] + d[perm[1]] + d[perm[2]];
    })

    code = code.replace(/\bivec3\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)/g, (_, x, y, z) => 
    {
        const args = [x.trim(), y.trim(), z.trim()];
        return `ivec3(${args[perm[0]]}, ${args[perm[1]]}, ${args[perm[2]]})`
        }
    )

    return code 
}

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
        float min4(vec4 v) { return min(min(min(v.r, v.g), v.b), v.a); }

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
            ivec3 coords = getCoords();
            CellValues c = getValues(coords);

            float xMin = getMinOnFaceX(c);
            float yMin = getMinOnFaceY(c);
            float zMin = getMinOnFaceZ(c);

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
            ivec3 coord = getCoords();

            CellValues c000 = getValues(coord + ivec3(0,0,0));
            CellValues c100 = getValues(coord + ivec3(1,0,0));
            CellValues c010 = getValues(coord + ivec3(0,1,0));
            CellValues c001 = getValues(coord + ivec3(0,0,1));

            float xMin = getMinOnFaceX(c000, c100);
            float yMin = getMinOnFaceY(c000, c010);
            float zMin = getMinOnFaceZ(c000, c001);

            setOutput(vec4(xMin, yMin, zMin, 0.0));
        }
        `
    }
}

class GPGPUMinimaMap2 implements GPGPUProgram 
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
        float min3(float a, float b, float c) { return min(min(a, b), c); }
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

        float getMinOnFaceX(CellValues c000, CellValues c100)
        {
            float m0 = min4(c000.v100, c000.v110, c000.v101, c000.v111);
            float m1 = min4(c000.v000, c000.v001, c000.v100, c000.v110);
            float m2 = min4(c100.v001, c100.v011, c100.v101, c100.v111);

            m1 = min(m1, avg3(c000.v000, c000.v001, c000.v100));
            m1 = min(m1, avg3(c000.v101, c000.v001, c000.v100));
            m1 = min(m1, c000.v101);

            m2 = min(m2, avg3(c100.v010, c100.v011, c100.v110));
            m2 = min(m2, avg3(c100.v111, c100.v011, c100.v110));
            m2 = min(m2, c100.v010);

            return max3(m0, m1, m2);
        }

        float getMinOnFaceY(CellValues c000, CellValues c010)
        {
            float m0 = min4(c000.v010, c000.v110, c000.v011, c000.v111);
            float m1 = min4(c000.v000, c000.v001, c000.v010, c000.v110);
            float m2 = min4(c010.v001, c010.v101, c010.v011, c010.v111);

            m1 = min(m1, avg3(c000.v000, c000.v001, c000.v010));
            m1 = min(m1, avg3(c000.v011, c000.v001, c000.v010));
            m1 = min(m1, c000.v011);

            m2 = min(m2, avg3(c010.v100, c010.v101, c010.v110));
            m2 = min(m2, avg3(c010.v111, c010.v101, c010.v110));
            m2 = min(m2, c010.v100);

            return max3(m0, m1, m2);
        }
            
        float getMinOnFaceZ(CellValues c000, CellValues c001)
        {
            float m0 = min4(c000.v001, c000.v101, c000.v011, c000.v111);
            float m1 = min4(c000.v000, c000.v100, c000.v010, c000.v110);
            float m2 = min4(c001.v001, c001.v101, c001.v011, c001.v111);

            m1 = min(m1, min4(c000.v000, c000.v100, c000.v001, c000.v101));
            m1 = min(m1, min4(c000.v000, c000.v010, c000.v001, c000.v011));
          
            m2 = min(m2, min4(c001.v010, c001.v110, c001.v011, c001.v111));
            m2 = min(m2, min4(c001.v001, c001.v101, c001.v011, c001.v111));
    
            return max3(m0, m1, m2);
        }

        void main()
        {
            ivec3 coord = getCoords();

            CellValues c000 = getValues(coord + ivec3(0,0,0));
            CellValues c100 = getValues(coord + ivec3(1,0,0));
            CellValues c010 = getValues(coord + ivec3(0,1,0));
            CellValues c001 = getValues(coord + ivec3(0,0,1));

            float xMin = getMinOnFaceX(c000, c100);
            float yMin = getMinOnFaceY(c000, c010);
            float zMin = getMinOnFaceZ(c000, c001);

            setOutput(vec4(xMin, yMin, zMin, 0.0));
        }
        `
    }
}

class GPGPUMinimaMap3 implements GPGPUProgram 
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
        float min3(float a, float b, float c) { return min(min(a, b), c); }
        float min4(float a, float b, float c, float d) { return min(min(min(a, b), c), d); }
        float min5(float a, float b, float c, float d, float e) { return min(min(min(min(a, b), c), d), e); }

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

        float getMinOnFaceX0(CellValues c) { return min4(c.v000, c.v010, c.v001, c.v011); }
        float getMinOnFaceY0(CellValues c) { return min4(c.v000, c.v100, c.v001, c.v101); }
        float getMinOnFaceZ0(CellValues c) { return min4(c.v000, c.v100, c.v010, c.v110); }
        float getMinOnFaceX1(CellValues c) { return min4(c.v100, c.v110, c.v101, c.v111); }
        float getMinOnFaceY1(CellValues c) { return min4(c.v010, c.v110, c.v011, c.v111); }
        float getMinOnFaceZ1(CellValues c) { return min4(c.v001, c.v101, c.v011, c.v111); }

        float getMinOnFaceX0TriangleYZ00(CellValues c) { return min5(c.v000, c.v010, c.v001, avg3(c.v000, c.v010, c.v011), avg3(c.v000, c.v001, c.v011)); }
        float getMinOnFaceX0TriangleYZ10(CellValues c) { return min5(c.v010, c.v000, c.v011, avg3(c.v010, c.v000, c.v001), avg3(c.v010, c.v011, c.v001)); }
        float getMinOnFaceX0TriangleYZ01(CellValues c) { return min5(c.v001, c.v000, c.v011, avg3(c.v001, c.v000, c.v010), avg3(c.v001, c.v011, c.v010)); }
        float getMinOnFaceX0TriangleYZ11(CellValues c) { return min5(c.v011, c.v010, c.v001, avg3(c.v011, c.v010, c.v000), avg3(c.v011, c.v001, c.v000)); }

        float getMinOnFaceX1TriangleYZ00(CellValues c) { return min5(c.v100, c.v110, c.v101, avg3(c.v100, c.v110, c.v111), avg3(c.v100, c.v101, c.v111)); }
        float getMinOnFaceX1TriangleYZ10(CellValues c) { return min5(c.v110, c.v100, c.v111, avg3(c.v110, c.v100, c.v101), avg3(c.v110, c.v111, c.v101)); }
        float getMinOnFaceX1TriangleYZ01(CellValues c) { return min5(c.v101, c.v100, c.v111, avg3(c.v101, c.v100, c.v110), avg3(c.v101, c.v111, c.v110)); }
        float getMinOnFaceX1TriangleYZ11(CellValues c) { return min5(c.v111, c.v110, c.v101, avg3(c.v111, c.v110, c.v100), avg3(c.v111, c.v101, c.v100)); }
        
        float getMinOnFaceY0TriangleXZ00(CellValues c) { return min5(c.v000, c.v100, c.v001, avg3(c.v000, c.v100, c.v101), avg3(c.v000, c.v001, c.v101)); }
        float getMinOnFaceY0TriangleXZ10(CellValues c) { return min5(c.v100, c.v000, c.v101, avg3(c.v100, c.v000, c.v001), avg3(c.v100, c.v101, c.v001)); }
        float getMinOnFaceY0TriangleXZ01(CellValues c) { return min5(c.v001, c.v000, c.v101, avg3(c.v001, c.v000, c.v100), avg3(c.v001, c.v101, c.v100)); }
        float getMinOnFaceY0TriangleXZ11(CellValues c) { return min5(c.v101, c.v100, c.v001, avg3(c.v101, c.v100, c.v000), avg3(c.v101, c.v001, c.v000)); }

        float getMinOnFaceY1TriangleXZ00(CellValues c) { return min5(c.v010, c.v110, c.v011, avg3(c.v010, c.v110, c.v111), avg3(c.v010, c.v011, c.v111)); }
        float getMinOnFaceY1TriangleXZ10(CellValues c) { return min5(c.v110, c.v010, c.v111, avg3(c.v110, c.v010, c.v011), avg3(c.v110, c.v111, c.v011)); }
        float getMinOnFaceY1TriangleXZ01(CellValues c) { return min5(c.v011, c.v010, c.v111, avg3(c.v011, c.v010, c.v110), avg3(c.v011, c.v111, c.v110)); }
        float getMinOnFaceY1TriangleXZ11(CellValues c) { return min5(c.v111, c.v110, c.v011, avg3(c.v111, c.v110, c.v010), avg3(c.v111, c.v011, c.v010)); }

        float getMinOnFaceZ0TriangleXY00(CellValues c) { return min5(c.v000, c.v100, c.v010, avg3(c.v000, c.v100, c.v110), avg3(c.v000, c.v010, c.v110)); }
        float getMinOnFaceZ0TriangleXY10(CellValues c) { return min5(c.v100, c.v000, c.v110, avg3(c.v100, c.v000, c.v010), avg3(c.v100, c.v110, c.v010)); }
        float getMinOnFaceZ0TriangleXY01(CellValues c) { return min5(c.v010, c.v000, c.v110, avg3(c.v010, c.v000, c.v100), avg3(c.v010, c.v110, c.v100)); }
        float getMinOnFaceZ0TriangleXY11(CellValues c) { return min5(c.v110, c.v100, c.v010, avg3(c.v110, c.v100, c.v000), avg3(c.v110, c.v010, c.v000)); }

        float getMinOnFaceZ1TriangleXY00(CellValues c) { return min5(c.v001, c.v101, c.v011, avg3(c.v001, c.v101, c.v111), avg3(c.v001, c.v011, c.v111)); }
        float getMinOnFaceZ1TriangleXY10(CellValues c) { return min5(c.v101, c.v001, c.v111, avg3(c.v101, c.v001, c.v011), avg3(c.v101, c.v111, c.v011)); }
        float getMinOnFaceZ1TriangleXY01(CellValues c) { return min5(c.v011, c.v001, c.v111, avg3(c.v011, c.v001, c.v101), avg3(c.v011, c.v111, c.v101)); }
        float getMinOnFaceZ1TriangleXY11(CellValues c) { return min5(c.v111, c.v101, c.v011, avg3(c.v111, c.v101, c.v001), avg3(c.v111, c.v011, c.v001)); }

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

        float getMinExitFaceX1(CellValues c11, CellValues c10)
        {
            float t10, t11;

            t10 = getMinOnFaceZ0(c10);
            t10 = max(getMinOnFaceY1TriangleXZ10(c10), t10);

            t11 = min(getMinOnFaceZ0(c11), t10);
            t11 = max(getMinOnFaceX1(c11), t11);

            return t11;
        }

        float getMinExitFaceY1(CellValues c11, CellValues c01)
        {
            float t01, t11;

            t01 = getMinOnFaceZ0(c01);
            t01 = max(getMinOnFaceX1TriangleYZ10(c01), t01);

            t11 = min(getMinOnFaceZ0(c11), t01);
            t11 = max(getMinOnFaceY1(c11), t11);

            return t11;
        }

        float getMinExitFaceZ1TriangleXY10(CellValues c11, CellValues c01, CellValues c10, CellValues c00)
        {
            float t00, t10, t01, t11;

            t00 = getMinOnFaceZ0(c00);

            t01 = max(getMinOnFaceY0TriangleXZ10(c01), t00);
            t01 = min(getMinOnFaceZ0TriangleXY10(c01), t01);
            t01 = max(getMinOnFaceX1TriangleYZ00(c01), t01);

            t10 = max(getMinOnFaceX0TriangleYZ10(c10), t00);
            t10 = min(getMinOnFaceZ0(c10), t10);
            t10 = max(getMinOnFaceY1(c10), t10);

            t11 = min3(getMinOnFaceZ0(c11), t10, t01);
            t11 = max(getMinOnFaceZ1TriangleXY10(c11), t11);

            return t11;
        }

        float getMinExitFaceZ1TriangleXY01(CellValues c11, CellValues c01, CellValues c10, CellValues c00)
        {
            float t00, t01, t10, t11;

            t00 = getMinOnFaceZ0(c00);

            t10 = max(getMinOnFaceX0TriangleYZ10(c10), t00);
            t10 = min(getMinOnFaceZ0TriangleXY01(c10), t10);
            t10 = max(getMinOnFaceY1TriangleXZ00(c10), t10);

            t01 = max(getMinOnFaceY0TriangleXZ10(c01), t00);
            t01 = min(getMinOnFaceZ0(c01), t01);
            t01 = max(getMinOnFaceX1(c01), t01);

            t11 = min3(getMinOnFaceZ0(c11), t01, t10);
            t11 = max(getMinOnFaceZ1TriangleXY01(c11), t11);

            return t11;
        }
    
        float getMinExitFaceZ1(CellValues c11, CellValues c01, CellValues c10, CellValues c00)
        {
            float t10 = getMinExitFaceZ1TriangleXY10(c11, c01, c10, c00);
            float t01 = getMinExitFaceZ1TriangleXY01(c11, c01, c10, c00);

            return min(t10, t01);
        }

        void main()
        {
            ivec3 coords = getCoords();

            CellValues c11 = getValues(coords - ivec3(0,0,0));
            CellValues c01 = getValues(coords - ivec3(1,0,0));
            CellValues c10 = getValues(coords - ivec3(0,1,0));
            CellValues c00 = getValues(coords - ivec3(1,1,0));

            float xMin = getMinExitFaceX1(c11, c01);
            float yMin = getMinExitFaceY1(c11, c10);
            float zMin = getMinExitFaceZ1(c11, c01, c10, c00);

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

        float min3(float a, float b, float c) { return min(min(a, b), c); }

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

        float getMinOnFaceZ(vec4 c111, vec4 c011, vec4 c101, vec4 c001, vec4 c110, vec4 c010, vec4 c100, vec4 c000)
        {
            float t00 = c000.z;

            float t01;
            t01 = max(c001.y, t00);
            t01 = min(c010.z, t01);
            t01 = max(c011.x, t01);

            float t10; 
            t10 = max(c001.x, t10);
            t10 = min(c100.z, t10);
            t10 = max(c101.y, t10);

            float t11;
            t11 = min3(c110.z, t10, t01);
            t11 = min(c110.z, t11);
            t11 = max(c111.z, t11);

            return t11;
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
            float wMax = 1.0;

            setOutput(vec4(xMax, yMax, zMax, wMax));
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
            ivec3 coords = getCoords();
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
            occlusion.w = all(occlusion.xyz) || occlusion.w;

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

export function computeOcclusionMap1(volumeMap: tf.Tensor3D) : tf.Tensor<tf.Rank>
{
    const minimaProgram = new GPGPUMinimaMap1(volumeMap.shape)
    const maximaProgram = new GPGPUMaximaMap1(volumeMap.shape)
    const updateProgram = new GPGPUUpdateSlice(volumeMap.shape)
    const occlusionProgram = new GPGPUOcclusionMap(volumeMap.shape)

    let minimaMap = runProgram(minimaProgram, [volumeMap])
    // console.log('minimaMap0', minimaMap.mean().dataSync())
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
    // console.log('minimaMap', minimaMap.mean().dataSync())

    const maximaMap = runProgram(maximaProgram, [volumeMap])
    const occlusionMap = runProgram(occlusionProgram, [minimaMap, maximaMap])
    tf.dispose([minimaMap, maximaMap])

    console.log('occlusionMap', occlusionMap.mean().dataSync())

    return occlusionMap as tf.Tensor
}

export function computeOmniOcclusionMap(volumeMap: tf.Tensor3D) : tf.Tensor<tf.Rank>
{
    const map = tf.transpose(volumeMap, [0, 1, 2])

    const posOcclusion = computeOcclusionMap1(map)
    const negOcclusion = tf.tidy(() => tf.reverse(computeOcclusionMap1(tf.reverse(map))))
    tf.dispose(map)

    const occlusionMap = tf.maximum(posOcclusion, negOcclusion)
    tf.dispose([posOcclusion, negOcclusion])

    console.log('occlusionMap', occlusionMap.mean().dataSync())

    return occlusionMap as tf.Tensor
}