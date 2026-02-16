import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { unstackPacked } from './unstack_packed_keepDims_webgl'
import { stackPacked } from './stack_packed_keepDims_webgl'

type Axis = 0 | 1 | 2
type Permute = [Axis, Axis, Axis]
type Reverse = Axis[]

class UnidirectionalMinimaMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(inputShape: [number, number, number], permute: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [inDepth, inHeight, inWidth] = inputShape
        const [outDepth, outHeight, outWidth] = inputShape.map(x => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]  

        const transformVoxelOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)

            for (const a of reverse) old[a] = 1 - old[a]
            
            return old.toReversed().join(',')
        }

        const transformCellOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)

            for (const a of reverse) old[a] = - old[a]
            
            return old.toReversed().join(',')
        }
    
        this.userCode = `
        const float minValue = 0.0; // uintBitsToFloat(0xFF800000u);
        const float maxValue = 1.0; // uintBitsToFloat(0x7F800000u);

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

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

        float avg3(float a, float b, float c) 
        { 
            return (a + b + c) * (1.0 / 3.0); 
        }

        float max3(float a, float b, float c) 
        {
            return max(max(a, b), c); 
        }

        float min4(float a, float b, float c, float d) 
        {
            return min(min(min(a, b), c), d); 
        }

        bool inBounds(ivec3 coords)
        {
            return all(greaterThanEqual(coords, minCoords)) && all(lessThanEqual(coords, maxCoords));
        }

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
            if (inBounds(vCoords)) 
                return getA(vCoords.z, vCoords.y, vCoords.x);
            else
                return minValue;
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

        bool isShadowed(CellValues c)
        {
            float m = 0.0;

            m = min(m, c.v000 - avg3(c.v001, c.v010, c.v100)); 
            m = min(m, c.v000 - avg3(c.v001, c.v010, c.v011)); 
            m = min(m, c.v000 - avg3(c.v001, c.v100, c.v101)); 
            m = min(m, c.v000 - avg3(c.v011, c.v101, c.v110)); 
            m = min(m, c.v100 - avg3(c.v101, c.v110, c.v111)); 
            m = min(m, c.v010 - avg3(c.v011, c.v110, c.v111)); 
            m = min(m, c.v000 - avg3(c.v000, c.v001, c.v010)); 
            m = min(m, c.v000 - avg3(c.v000, c.v001, c.v100)); 
            m = min(m, c.v100 - avg3(c.v100, c.v101, c.v110)); 
            m = min(m, c.v010 - avg3(c.v010, c.v011, c.v110)); 
            m = min(m, c.v000 - c.v001); 
            m = min(m, c.v000 - c.v011); 
            m = min(m, c.v000 - c.v101); 
            m = min(m, c.v000 - c.v111); 
            m = min(m, c.v100 - c.v101); 
            m = min(m, c.v100 - c.v111); 
            m = min(m, c.v010 - c.v011); 
            m = min(m, c.v010 - c.v111); 
            m = min(m, c.v110 - c.v111); 

            return (m >= 0.0);
        }

        bool isShadowed2(CellValues c)
        {
            float m = 0.0;

            m = min(m, c.v111 - avg3(c.v110, c.v101, c.v011)); 
            m = min(m, c.v111 - avg3(c.v110, c.v101, c.v100)); 
            m = min(m, c.v111 - avg3(c.v110, c.v011, c.v010)); 
            m = min(m, c.v111 - avg3(c.v100, c.v010, c.v001)); 
            m = min(m, c.v011 - avg3(c.v010, c.v001, c.v000)); 
            m = min(m, c.v101 - avg3(c.v100, c.v001, c.v000)); 
            m = min(m, c.v111 - avg3(c.v111, c.v110, c.v101)); 
            m = min(m, c.v111 - avg3(c.v111, c.v110, c.v011)); 
            m = min(m, c.v011 - avg3(c.v011, c.v010, c.v001)); 
            m = min(m, c.v101 - avg3(c.v101, c.v100, c.v001)); 
            m = min(m, c.v111 - c.v110); 
            m = min(m, c.v111 - c.v100); 
            m = min(m, c.v111 - c.v010); 
            m = min(m, c.v111 - c.v000); 
            m = min(m, c.v011 - c.v010); 
            m = min(m, c.v011 - c.v000); 
            m = min(m, c.v101 - c.v100); 
            m = min(m, c.v101 - c.v000); 
            m = min(m, c.v001 - c.v000); 

            return (m >= 0.0);
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

            // if (isShadowed(c000))
            //     setOutput(vec4(minValue, minValue, minValue, 1.0));
            // else
            //     setOutput(vec4(xMin, yMin, zMin, 0.0));

            setOutput(vec4(minValue, minValue, minValue, isShadowed(c000) || isShadowed2(c000)));
        }
        `
    }
}

class BidirectionalMinimaMap implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(inputShape: [number, number, number], permute: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [inDepth, inHeight, inWidth] = inputShape
        const [outDepth, outHeight, outWidth] = inputShape.map(x => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]  

        const transformVoxelOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)

            for (const a of reverse) old[a] = 1 - old[a]
            
            return old.toReversed().join(',')
        }

        const transformCellOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)

            for (const a of reverse) old[a] = - old[a]
            
            return old.toReversed().join(',')
        }
    
        this.userCode = `
        const float minValue = 0.0; // uintBitsToFloat(0xFF800000u);
        const float maxValue = 1.0; // uintBitsToFloat(0x7F800000u);

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

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

        float avg3(float a, float b, float c) 
        { 
            return (a + b + c) * (1.0 / 3.0); 
        }

        float max3(float a, float b, float c) 
        {
            return max(max(a, b), c); 
        }

        float min4(float a, float b, float c, float d) 
        {
            return min(min(min(a, b), c), d); 
        }

        bool inBounds(ivec3 coords)
        {
            return all(greaterThanEqual(coords, minCoords)) && all(lessThanEqual(coords, maxCoords));
        }

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
            if (inBounds(vCoords)) 
                return getA(vCoords.z, vCoords.y, vCoords.x);
            else
                return minValue;
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

        bool isShadowed(CellValues c)
        {
            float m = 0.0;

            m = min(m, c.v000 - avg3(c.v001, c.v010, c.v100)); 
            m = min(m, c.v000 - avg3(c.v001, c.v010, c.v011)); 
            m = min(m, c.v000 - avg3(c.v001, c.v100, c.v101)); 
            m = min(m, c.v000 - avg3(c.v011, c.v101, c.v110)); 
            m = min(m, c.v100 - avg3(c.v101, c.v110, c.v111)); 
            m = min(m, c.v010 - avg3(c.v011, c.v110, c.v111)); 
            m = min(m, c.v000 - avg3(c.v000, c.v001, c.v010)); 
            m = min(m, c.v000 - avg3(c.v000, c.v001, c.v100)); 
            m = min(m, c.v100 - avg3(c.v100, c.v101, c.v110)); 
            m = min(m, c.v010 - avg3(c.v010, c.v011, c.v110)); 
            m = min(m, c.v000 - c.v001); 
            m = min(m, c.v000 - c.v011); 
            m = min(m, c.v000 - c.v101); 
            m = min(m, c.v000 - c.v111); 
            m = min(m, c.v100 - c.v101); 
            m = min(m, c.v100 - c.v111); 
            m = min(m, c.v010 - c.v011); 
            m = min(m, c.v010 - c.v111); 
            m = min(m, c.v110 - c.v111); 

            return (m >= 0.0);
        }

        bool isMasked(ivec3 cCoords)
        {
            return (getB(cCoords.z, cCoords.y, cCoords.x) > 0.5);
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

            if (isMasked(cCoords))
                setOutput(vec4(minValue, minValue, minValue, 1.0));
            else
                setOutput(vec4(xMin, yMin, zMin, isShadowed(cCoords)));
        }
        `
    }
}

class MaskUnidirectionalMinimaMap implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(outputShape: [number, number, number, 2, 2]) 
    {
        this.outputShape = outputShape  
        this.userCode = `
        const float minValue = 0.0; // uintBitsToFloat(0xFF800000u);
        const float maxValue = 1.0; // uintBitsToFloat(0x7F800000u);

        float min3(float a, float b, float c) 
        { 
            return min(min(a, b), c); 
        }

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
            return getA(cCoords.z, cCoords.y, cCoords.x, 0, 0);
        }

        float getB(ivec3 cCoords)
        {
            vec4 v = getB(cCoords.z, cCoords.y, cCoords.x);

            bool yEven = (cCoords.y & 1) == 0;
            bool xEven = (cCoords.x & 1) == 0;

            if (yEven) 
                return xEven ? v.r : v.g;
            else 
                return xEven ? v.b : v.a;
        }

        bool isMasked(ivec3 cCoords)
        {
            return (getB(cCoords) > 0.5);
        }
  
        void main()
        {
            ivec3 cCoords = getOutCoords();

            if (isMasked(cCoords)) 
                setOutput(vec4(minValue, minValue, minValue, 1.0));
            else
                setOutput(getA(cCoords));
        }
        `
    }
}

class PropagateUnidirectionalMinimaSlices implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(outputShape: [number, number, number, 2, 2], permute: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [outDepth, outHeight, outWidth] = outputShape.slice(0, 3)
        this.outputShape = outputShape 
        
        const transformCellOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)

            for (const a of reverse) old[a] = - old[a]

            const axis = permute[0]
            old[axis] = 0
            
            return old.toReversed().join(',')
        }

        this.userCode = `
        const float minValue = 0.0; // uintBitsToFloat(0xFF800000u);
        const float maxValue = 1.0; // uintBitsToFloat(0x7F800000u);

        const ivec3 minCCoords = ivec3(0);
        const ivec3 maxCCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        float min3(float a, float b, float c) 
        { 
            return min(min(a, b), c); 
        }

        bool inBounds(ivec3 cCoords)
        {
            return all(greaterThanEqual(cCoords, minCCoords)) && all(lessThanEqual(cCoords, maxCCoords));
        }

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
            if (inBounds(cCoords)) 
                return getA(cCoords.z, cCoords.y, cCoords.x, 0, 0);
            else
                return vec4(minValue, minValue, minValue, 0.0);
        }

        vec4 getB(ivec3 cCoords)
        {
            if (inBounds(cCoords)) 
                return getB(cCoords.z, cCoords.y, cCoords.x, 0, 0);
            else
                return vec4(minValue, minValue, minValue, 0.0);    
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

class PropagateUnidirectionalMinimaMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(outputShape: [number, number, number, 2, 2], permute: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [outDepth, outHeight, outWidth] = outputShape.slice(0,3)
        this.outputShape = outputShape  

        const transformCellOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)

            for (const a of reverse) old[a] = - old[a]
            
            return old.toReversed().join(',')
        }

        this.userCode = `
        const float minValue = 0.0; // uintBitsToFloat(0xFF800000u);
        const float maxValue = 1.0; // uintBitsToFloat(0x7F800000u);

        const ivec3 minCCoords = ivec3(0);
        const ivec3 maxCCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        float min3(float a, float b, float c) 
        { 
            return min(min(a, b), c); 
        }

        bool inBounds(ivec3 cCoords)
        {
            return all(greaterThanEqual(cCoords, minCCoords)) && all(lessThanEqual(cCoords, maxCCoords));
        }

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
            if (inBounds(cCoords)) 
                return getA(cCoords.z, cCoords.y, cCoords.x, 0, 0);
            else
                return vec4(minValue, minValue, minValue, 0.0);
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

    constructor(inputShape: [number, number, number], permute: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [inDepth, inHeight, inWidth] = inputShape
        const [outDepth, outHeight, outWidth] = inputShape.map(x => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]     

        const transformVoxelOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)

            for (const a of reverse) old[a] = 1 - old[a]
            
            return old.toReversed().join(',')
        }

        this.userCode = `
        const float minValue = 0.0; // uintBitsToFloat(0xFF800000u);
        const float maxValue = 1.0; // uintBitsToFloat(0x7F800000u);

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

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

        float avg3(float a, float b, float c) 
        { 
            return (a + b + c) * (1.0 / 3.0); 
        }

        float max4(float a, float b, float c, float d) 
        { 
            return max(max(max(a, b), c), d); 
        }

        bool inBounds(ivec3 coords)
        {
            return all(greaterThanEqual(coords, minCoords)) && all(lessThanEqual(coords, maxCoords));
        }

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
            if (inBounds(vCoords)) 
                return getA(vCoords.z, vCoords.y, vCoords.x);
            else
                return minValue;
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
            float m = minValue;

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
            float m = minValue;

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
            float m = minValue;

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

        void main()
        {
            ivec3 cCoords = getOutCoords();
            CellValues c = getValues(cCoords);

            float xMax = getMaxOnFaceX(c);
            float yMax = getMaxOnFaceY(c);
            float zMax = getMaxOnFaceZ(c);

            setOutput(vec4(xMax, yMax, zMax, 0.5));
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

    constructor(outputShape: [number, number, number], permute: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [outDepth, outHeight, outWidth] = outputShape
        this.outputShape = outputShape

        const transformCellOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)

            for (const a of reverse) old[a] = - old[a]
            
            return old.toReversed().join(',')
        }

        this.userCode = `
        const float minValue = 0.0; // uintBitsToFloat(0xFF800000u);
        const float maxValue = 1.0; // uintBitsToFloat(0x7F800000u);

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        bool inBounds(ivec3 coords)
        {
            return all(greaterThanEqual(coords, minCoords)) && all(lessThanEqual(coords, maxCoords));
        }

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
            if (inBounds(cCoords))
                return getA(cCoords.z, cCoords.y, cCoords.x, 0, 0);
            else 
                return vec4(minValue, minValue, minValue, 0.0);
        }

        vec4 getB(ivec3 cCoords)
        {
            if (inBounds(cCoords))
                return getB(cCoords.z, cCoords.y, cCoords.x, 0, 0);
            else 
                return vec4(maxValue, maxValue, maxValue, 0.0);
        }

        bool isShadowed(vec4 minValues, vec4 maxValues)
        {
            bvec4 conditions = greaterThan(minValues, maxValues);
            return conditions.w;
        }
                
        void main()
        {
            ivec3 cCoords = getOutCoords();

            vec4 b111 = getB(getCCoords(cCoords, ${transformCellOffset(-0,-0,-0)}));
            vec4 a111 = getA(getCCoords(cCoords, ${transformCellOffset(-0,-0,-0)}));
            vec4 a011 = getA(getCCoords(cCoords, ${transformCellOffset(-1,-0,-0)}));
            vec4 a101 = getA(getCCoords(cCoords, ${transformCellOffset(-0,-1,-0)}));
            vec4 a110 = getA(getCCoords(cCoords, ${transformCellOffset(-0,-0,-1)}));

            vec4 minValues = vec4(a011.x, a101.y, a110.z, a111.w);
            vec4 maxValues = vec4(b111.x, b111.y, b111.z, b111.w);

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

// encapsulation functions

async function propagateUnidirectionalMinimaMapAsync(
    minima: tf.Tensor5D, 
    permute: Permute, 
    reverse: Reverse
): Promise<tf.Tensor5D>
{
    const shape = minima.shape as [number, number, number, 2, 2]
    const program = new PropagateUnidirectionalMinimaMap(shape, permute, reverse)

    const length = shape[permute[0]]

    for (let i = 1; i < length; i++)
    {
        const tensor = runWebGLProgram(program, [minima], 'float32', [], true)
        tf.dispose(minima)
        minima = tensor as tf.Tensor5D

        await tf.nextFrame()
    }
    
    return minima
}

async function computePropagatedUnidirectionalMinimaMapAsync(
    volume: tf.Tensor3D, 
    mask: tf.Tensor3D | undefined, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
): Promise<tf.Tensor5D>
{
    const program = new UnidirectionalMinimaMap(volume.shape, permute, reverse)
    let minima = runWebGLProgram(program, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logTensor('minimaStart', minima)

    if (mask instanceof tf.Tensor)
    {
        const shape = minima.shape as [number, number, number, 2, 2]
        const maskProgram = new MaskUnidirectionalMinimaMap(shape)
        const masked = runWebGLProgram(maskProgram, [minima, mask], 'float32', [], true) as tf.Tensor5D
        minima.dispose()
        minima = masked
        if (verbose) logTensor('minimaMasked', minima)
    }

    minima = await propagateUnidirectionalMinimaMapAsync(minima, permute, reverse) as tf.Tensor5D
    if (verbose) logTensor('minimaPropagated', minima)

    return minima
}

function propagateUnidirectionalMinimaMap(
    minima: tf.Tensor5D, 
    permute: Permute, 
    reverse: Reverse
): tf.Tensor5D
{
    const axis = permute[0]
    const slices = unstackPacked(minima, axis) 
    minima.dispose()

    const shape = slices[0].shape as [number, number, number, 2, 2]
    const program = new PropagateUnidirectionalMinimaSlices(shape, permute, reverse)

    const toReverse = reverse.includes(axis)
    if (toReverse) slices.reverse()
        
    for (let i = 1; i < slices.length; i++)
    {
        const slice = runWebGLProgram(program, [slices[i], slices[i-1]], 'float32', [[i]], true)
        tf.dispose(slices[i])
        slices[i] = slice
    }

    if (toReverse) slices.reverse()

    minima = stackPacked(slices, axis) as tf.Tensor5D 
    tf.dispose(slices)

    return minima
}

function computePropagatedUnidirectionalMinimaMap(
    volume: tf.Tensor3D, 
    mask: tf.Tensor3D | undefined, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
): tf.Tensor5D
{
    const program = new UnidirectionalMinimaMap(volume.shape, permute, reverse)
    let minima = runWebGLProgram(program, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logTensor('minimaStart', minima)

    if (mask instanceof tf.Tensor)
    {
        const shape = minima.shape as [number, number, number, 2, 2]
        const maskProgram = new MaskUnidirectionalMinimaMap(shape)
        const masked = runWebGLProgram(maskProgram, [minima, mask], 'float32', [], true) as tf.Tensor5D
        minima.dispose()
        minima = masked
        if (verbose) logTensor('minimaMasked', minima)
    }

    minima = propagateUnidirectionalMinimaMap(minima, permute, reverse) as tf.Tensor5D
    if (verbose) logTensor('minimaPropagated', minima)

    return minima
}

function computeUnidirectionalMaximaMap(
    volume: tf.Tensor3D, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
) : tf.Tensor5D
{
    const program = new UnidirectionalMaximaMap(volume.shape, permute, reverse)
    const maxima = runWebGLProgram(program, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logTensor('maxima', maxima)

    return maxima 
}

// sync functions 

export function computeUnidirectionalShadowMap(
    volume: tf.Tensor3D, 
    mask: tf.Tensor3D | undefined, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
) : tf.Tensor3D
{
    const minima = computePropagatedUnidirectionalMinimaMap(volume, mask, permute, reverse, verbose)
    const maxima = computeUnidirectionalMaximaMap(volume, permute, reverse, verbose)

    const shape = minima.shape.slice(0,3) as [number, number, number]
    const program = new UnidirectionalShadowMap(shape, permute, reverse)
    const shadows = runWebGLProgram(program, [minima, maxima], 'float32', [], true) as tf.Tensor3D
    tf.dispose([minima, maxima])
    if (verbose) logTensor('shadows', shadows)

    return shadows as tf.Tensor3D
}

// export function computeBidirectionalShadowMap(
//     volume: tf.Tensor3D, 
//     permute: Permute, 
//     reverse: Reverse, 
//     verbose: boolean = false
// ) : tf.Tensor3D
// {
//     const shadows = computeUnidirectionalShadowMap(volume, undefined, permute, reverse)
//     if (verbose) logTensor('shadows', shadows)
        
//     const invShadows = computeUnidirectionalShadowMap(volume, shadows, permute, complementReverse(reverse))
//     if (verbose) logTensor('invShadows', invShadows)

//     const or = new BidirectionalShadowMap(shadows.shape)
//     const biShadows = runWebGLProgram(or, [shadows, invShadows], 'float32', [], true) as tf.Tensor3D
//     tf.dispose([shadows, invShadows])
//     if (verbose) logTensor('biShadows', biShadows)

//     return biShadows as tf.Tensor3D
// }

export function computeBidirectionalShadowMap(
    volume: tf.Tensor3D, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
) : tf.Tensor3D
{
    const invShadows = computeUnidirectionalShadowMap(volume, undefined, permute, complementReverse(reverse))
    if (verbose) logTensor('invShadows', invShadows)
        
    const shadows = computeUnidirectionalShadowMap(volume, invShadows, permute, reverse)
    tf.dispose(invShadows)
    if (verbose) logTensor('shadows', shadows)

    return shadows as tf.Tensor3D
}

export function computeAnisotropicBidirectionalShadowMap(
    volume: tf.Tensor3D, 
    permute: Permute, 
    verbose: boolean = false
) : tf.Tensor3D
{
    const reverseA = permute.slice(1, 1) as Reverse
    const reverseB = permute.slice(1, 2) as Reverse
    const reverseC = permute.slice(2, 3) as Reverse
    const reverseD = permute.slice(1, 3) as Reverse

    const shadowMaps = [
        computeBidirectionalShadowMap(volume, permute, reverseA),
        computeBidirectionalShadowMap(volume, permute, reverseB),
        computeBidirectionalShadowMap(volume, permute, reverseC),
        computeBidirectionalShadowMap(volume, permute, reverseD),
    ]

    const program = new AnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(program, shadowMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logAnisotropicBidirectionalShadowMaps(shadowMap)

    return shadowMap 
}

export function computeExtendedAnisotropicBidirectionalShadowMap(
    volume: tf.Tensor3D, 
    verbose: boolean = false
) : tf.Tensor3D
{
    const permuteX = [2,1,0] as Permute
    const permuteY = [1,2,0] as Permute
    const permuteZ = [0,1,2] as Permute

    const shadowMaps = [
        computeAnisotropicBidirectionalShadowMap(volume, permuteX),
        computeAnisotropicBidirectionalShadowMap(volume, permuteY),
        computeAnisotropicBidirectionalShadowMap(volume, permuteZ),
    ]

    const program = new ExtendedAnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(program, shadowMaps, 'int32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logExtendedAnisotropicBidirectionalShadowMaps(shadowMap)

    return shadowMap 
}

// async functions 

export async function computeUnidirectionalShadowMapAsync(
    volume: tf.Tensor3D, 
    mask: tf.Tensor3D | undefined, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
) : Promise<tf.Tensor3D>
{
    const minima = await computePropagatedUnidirectionalMinimaMapAsync(volume, mask, permute, reverse, verbose)
    const maxima = computeUnidirectionalMaximaMap(volume, permute, reverse, verbose)

    const shape = minima.shape.slice(0,3) as [number, number, number]
    const program = new UnidirectionalShadowMap(shape, permute, reverse)
    const shadows = runWebGLProgram(program, [minima, maxima], 'float32', [], true) as tf.Tensor3D
    tf.dispose([minima, maxima])
    if (verbose) logTensor('shadows', shadows)

    return shadows as tf.Tensor3D
}

export async function computeBidirectionalShadowMapAsync(
    volume: tf.Tensor3D, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
) : Promise<tf.Tensor3D>
{
    const shadows = await computeUnidirectionalShadowMapAsync(volume, undefined, permute, reverse)
    if (verbose) logTensor('shadows', shadows)
        
    const invShadows = await computeUnidirectionalShadowMapAsync(volume, shadows, permute, complementReverse(reverse))
    if (verbose) logTensor('invShadows', invShadows)

    const or = new BidirectionalShadowMap(shadows.shape)
    const biShadows = runWebGLProgram(or, [shadows, invShadows], 'float32', [], true) as tf.Tensor3D
    tf.dispose([shadows, invShadows])
    if (verbose) logTensor('biShadows', biShadows)

    return biShadows as tf.Tensor3D
}

export async function computeAnisotropicBidirectionalShadowMapAsync(
    volume: tf.Tensor3D, 
    permute: Permute, 
    verbose: boolean = false
) : Promise<tf.Tensor3D>
{
    const reverseA = permute.slice(1, 1) as Reverse
    const reverseB = permute.slice(1, 2) as Reverse
    const reverseC = permute.slice(2, 3) as Reverse
    const reverseD = permute.slice(1, 3) as Reverse

    const shadowMaps = [
        await computeBidirectionalShadowMapAsync(volume, permute, reverseA),
        await computeBidirectionalShadowMapAsync(volume, permute, reverseB),
        await computeBidirectionalShadowMapAsync(volume, permute, reverseC),
        await computeBidirectionalShadowMapAsync(volume, permute, reverseD),
    ]

    const program = new AnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(program, shadowMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logAnisotropicBidirectionalShadowMaps(shadowMap)

    return shadowMap 
}

export async function computeExtendedAnisotropicBidirectionalShadowMapAsync(
    volume: tf.Tensor3D, 
    verbose: boolean = false
) : Promise<tf.Tensor3D>
{
    const permuteX = [2,1,0] as Permute
    const permuteY = [1,2,0] as Permute
    const permuteZ = [0,1,2] as Permute

    const shadowMaps = [
        await computeAnisotropicBidirectionalShadowMapAsync(volume, permuteX),
        await computeAnisotropicBidirectionalShadowMapAsync(volume, permuteY),
        await computeAnisotropicBidirectionalShadowMapAsync(volume, permuteZ),
    ]

    const program = new ExtendedAnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(program, shadowMaps, 'int32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logExtendedAnisotropicBidirectionalShadowMaps(shadowMap)

    return shadowMap 
}

// base comparison 

export function computeUnidirectionalShadowMapBase(
    volume: tf.Tensor3D, 
    mask: tf.Tensor3D | undefined, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
) : tf.Tensor3D
{
    const reversed = volume.reverse(reverse) as tf.Tensor3D
    const transposed = reversed.transpose(permute) as tf.Tensor3D
    tf.dispose(reversed)

    const minima = computePropagatedUnidirectionalMinimaMap(transposed, mask, [0,1,2], [], verbose)
    const maxima = computeUnidirectionalMaximaMap(transposed, [0,1,2], [], verbose)
    tf.dispose(transposed)

    const shape = minima.shape.slice(0,3) as [number, number, number]
    const program = new UnidirectionalShadowMap(shape, [0,1,2], [])
    const shadows = runWebGLProgram(program, [minima, maxima], 'float32', [], true) as tf.Tensor3D
    tf.dispose([minima, maxima])
    if (verbose) logTensor('shadows', shadows)

    const untransposed = shadows.transpose(inversePermutation(permute))
    tf.dispose(shadows)
    const unreversed = untransposed.reverse(reverse)
    tf.dispose(untransposed)

    return unreversed as tf.Tensor3D
}

export function computeBidirectionalShadowMapBase(
    volume: tf.Tensor3D, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
) : tf.Tensor3D
{
    const shadows = computeUnidirectionalShadowMapBase(volume, undefined, permute, reverse)
    if (verbose) logTensor('shadows', shadows)
        
    const invShadows = computeUnidirectionalShadowMapBase(volume, shadows, permute, complementReverse(reverse))
    if (verbose) logTensor('invShadows', invShadows)

    const or = new BidirectionalShadowMap(shadows.shape)
    const biShadows = runWebGLProgram(or, [shadows, invShadows], 'float32', [], true) as tf.Tensor3D
    tf.dispose([shadows, invShadows])
    if (verbose) logTensor('biShadows', biShadows)

    return biShadows as tf.Tensor3D
}

export function computeAnisotropicBidirectionalShadowMapBase(
    volume: tf.Tensor3D, 
    permute: Permute, 
    verbose: boolean = false
) : tf.Tensor3D
{
    const reverseA = permute.slice(1, 1) as Reverse
    const reverseB = permute.slice(1, 2) as Reverse
    const reverseC = permute.slice(2, 3) as Reverse
    const reverseD = permute.slice(1, 3) as Reverse

    const shadowMaps = [
        computeBidirectionalShadowMapBase(volume, permute, reverseA),
        computeBidirectionalShadowMapBase(volume, permute, reverseB),
        computeBidirectionalShadowMapBase(volume, permute, reverseC),
        computeBidirectionalShadowMapBase(volume, permute, reverseD),
    ]

    const program = new AnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(program, shadowMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logAnisotropicBidirectionalShadowMaps(shadowMap)

    return shadowMap 
}

export function computeExtendedAnisotropicBidirectionalShadowMapBase(
    volume: tf.Tensor3D, 
    verbose: boolean = false
) : tf.Tensor3D
{
    const permuteX = [2,1,0] as Permute
    const permuteY = [1,2,0] as Permute
    const permuteZ = [0,1,2] as Permute

    const shadowMaps = [
        computeAnisotropicBidirectionalShadowMapBase(volume, permuteX),
        computeAnisotropicBidirectionalShadowMapBase(volume, permuteY),
        computeAnisotropicBidirectionalShadowMapBase(volume, permuteZ),
    ]

    const program = new ExtendedAnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadowMap = runWebGLProgram(program, shadowMaps, 'int32', [], true) as tf.Tensor3D
    tf.dispose(shadowMaps)

    if (verbose) logExtendedAnisotropicBidirectionalShadowMaps(shadowMap)

    return shadowMap 
}

// debug

export function computeExtendedAnisotropicBidirectionalShadowMapDebug(
    volume: tf.Tensor3D, 
    verbose: boolean = false
) : tf.Tensor3D
{
    let o1,o2,o3,o4,ox,oy,oz

    // let t = computeUnidirectionalShadowMap(volume, [0,1,2], [])
    // let t = computeUnidirectionalShadowMap(volume, [0,1,2], [0,1,2])
    let t = computeUnidirectionalShadowMap(volume, undefined, [0,1,2], [], true)

    o1 = tf.onesLike(t) // computeBidirectionalShadowMap(volume, [2,1,0], [   ])
    o2 = tf.onesLike(t) // computeBidirectionalShadowMap(volume, [2,1,0], [  1])
    o3 = tf.onesLike(t) // computeBidirectionalShadowMap(volume, [2,1,0], [  0])
    o4 = tf.onesLike(t) // computeBidirectionalShadowMap(volume, [2,1,0], [1,0])

    ox = runWebGLProgram(new AnisotropicBidirectionalShadowMap(t.shape), [o1,o2,o3,o4], 'float32', [], true) as tf.Tensor3D
    tf.dispose([o1,o2,o3,o4])
   
    o1 = tf.onesLike(t) // computeBidirectionalShadowMap(volume, [1,2,0], [   ])
    o2 = tf.onesLike(t) // computeBidirectionalShadowMap(volume, [1,2,0], [  2])
    o3 = tf.onesLike(t) // computeBidirectionalShadowMap(volume, [1,2,0], [  0])
    o4 = tf.onesLike(t) // computeBidirectionalShadowMap(volume, [1,2,0], [2,0])

    oy = runWebGLProgram(new AnisotropicBidirectionalShadowMap(t.shape), [o1,o2,o3,o4], 'float32', [], true) as tf.Tensor3D
    tf.dispose([o1,o2,o3,o4])

    o1 = tf.clone(t) // computeBidirectionalShadowMap(volume, [0,1,2], [   ])
    o2 = tf.onesLike(t) // computeBidirectionalShadowMap(volume, [0,1,2], [  1])
    o3 = tf.onesLike(t) // computeBidirectionalShadowMap(volume, [0,1,2], [  2])
    o4 = tf.onesLike(t) // computeBidirectionalShadowMap(volume, [0,1,2], [1,2])

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

function inversePermutation(permute: Permute): Permute
{
    const inv = new Array<number>(permute.length)
    for (let i = 0; i < permute.length; i++) 
    {
        inv[permute[i]] = i
    }

    return inv as Permute
}

function applyPermutation(newOffset: [number, number, number], permute: Permute): [number, number, number] 
{
    const oldOffset: [number, number, number] = [0, 0, 0]

    oldOffset[permute[0]] = newOffset[0]
    oldOffset[permute[1]] = newOffset[1]
    oldOffset[permute[2]] = newOffset[2]
    
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
