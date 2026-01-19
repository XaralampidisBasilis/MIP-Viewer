import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class GPGPUStartPropagationMap implements GPGPUProgram 
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
        this.outputShape = [outDepth, outHeight, outWidth]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float mmin(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        ivec3 getVoxelCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        vec4 getPackedValues(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            bool insideWidth  = (safeCoords.x < maxCoords.x);
            bool insideHeight = (safeCoords.y < maxCoords.y);

            vec4 packedValues = getA(safeCoords.z, safeCoords.y, safeCoords.x);
            packedValues.ga = insideWidth ? packedValues.ga : packedValues.rb;
            packedValues.ba = insideHeight ? packedValues.ba : packedValues.rg;

            return packedValues;
        }

        void main()
        {
            ivec3 voxelCoords = getVoxelCoords();

            vec4 F000 = getPackedValues(voxelCoords + ivec3(0,0,0));
            vec4 F100 = getPackedValues(voxelCoords + ivec3(2,0,0));
            vec4 F010 = getPackedValues(voxelCoords + ivec3(0,2,0));
            vec4 F110 = getPackedValues(voxelCoords + ivec3(2,2,0));

            vec4 L000;
            L000.r = mmin(F000.r, F000.g, F000.b, F000.a);
            L000.g = mmin(F100.r, F000.g, F100.b, F000.a);
            L000.b = mmin(F010.r, F010.g, F000.b, F000.a);
            L000.a = mmin(F110.r, F010.g, F100.b, F000.a);

            setOutput(L000);
        }
        `
    }
}

class GPGPUBackPropagationMap implements GPGPUProgram 
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
        this.outputShape = [inDepth, inHeight, inWidth]  
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth}, ${inHeight}, ${inDepth});

        float mmin(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        ivec3 getVoxelCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        vec4 getPackedValues(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            bool insideWidth  = (safeCoords.x < maxCoords.x);
            bool insideHeight = (safeCoords.y < maxCoords.y);

            vec4 packedValues = getA(safeCoords.z, safeCoords.y, safeCoords.x);
            packedValues.ga = insideWidth ? packedValues.ga : packedValues.rb;
            packedValues.ba = insideHeight ? packedValues.ba : packedValues.rg;

            return packedValues;
        }

        void main()
        {
            ivec3 voxelCoords = getVoxelCoords();

            vec4 L000 = getPackedValues(voxelCoords - ivec3(0,0,0));
            vec4 L001 = getPackedValues(voxelCoords - ivec3(0,0,1));
            vec4 L101 = getPackedValues(voxelCoords - ivec3(2,0,1));
            vec4 L011 = getPackedValues(voxelCoords - ivec3(0,2,1));
            vec4 L111 = getPackedValues(voxelCoords - ivec3(2,2,1));

            L000.r = max(L000.r, mmin(L001.r, L101.g, L011.b, L111.a));
            L000.g = max(L000.g, mmin(L001.r, L001.g, L011.b, L011.a));
            L000.b = max(L000.b, mmin(L001.r, L101.g, L001.b, L101.a));
            L000.a = max(L000.a, mmin(L001.r, L001.g, L001.b, L001.a));

            setOutput(L000);
        }
        `
    }
}


class GPGPUFrontPropagationMap implements GPGPUProgram 
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
        this.outputShape = [inDepth, inHeight, inWidth]  
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth}, ${inHeight}, ${inDepth});

        float mmin(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        ivec3 getVoxelCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        vec4 getPackedValues(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            bool insideWidth  = (safeCoords.x < maxCoords.x);
            bool insideHeight = (safeCoords.y < maxCoords.y);

            vec4 packedValues = getA(safeCoords.z, safeCoords.y, safeCoords.x);
            packedValues.ga = insideWidth ? packedValues.ga : packedValues.rb;
            packedValues.ba = insideHeight ? packedValues.ba : packedValues.rg;

            return packedValues;
        }

        void main()
        {
            ivec3 voxelCoords = getVoxelCoords();

            vec4 L000 = getPackedValues(voxelCoords + ivec3(0,0,0));
            vec4 L001 = getPackedValues(voxelCoords + ivec3(0,0,1));
            vec4 L101 = getPackedValues(voxelCoords + ivec3(2,0,1));
            vec4 L011 = getPackedValues(voxelCoords + ivec3(0,2,1));
            vec4 L111 = getPackedValues(voxelCoords + ivec3(2,2,1));

            L000.r = max(L000.r, mmin(L001.r, L001.g, L001.b, L001.a));
            L000.g = max(L000.g, mmin(L101.r, L001.g, L101.b, L001.a));
            L000.b = max(L000.b, mmin(L011.r, L011.g, L001.b, L001.a));
            L000.a = max(L000.a, mmin(L111.r, L011.g, L101.b, L001.a));

            setOutput(L000);
        }
        `
    }
}

class GPGPUBackOcclusionMap implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = false
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

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float mmax(float a1, float a2, float a3, float a4)
        {
            return max(max(max(a1, a2), a3), a4);
        }

        float mmin(float a1, float a2, float a3, float a4, float a5, float a6, float a7)
        {
            return min(min(min(min(min(min(a1, a2), a3), a4), a5), a6), a7);
        }
   
        ivec3 getCellCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getVoxelValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        float getPropagatedValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getB(safeCoords.z, safeCoords.y, safeCoords.x);
        }    

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            ivec3 voxelCoords = cellCoords - 1;
            
            float f000 = getVoxelValue(voxelCoords + ivec3(0,0,0));
            float f100 = getVoxelValue(voxelCoords + ivec3(1,0,0));
            float f010 = getVoxelValue(voxelCoords + ivec3(0,1,0));
            float f001 = getVoxelValue(voxelCoords + ivec3(0,0,1));
            float f011 = getVoxelValue(voxelCoords + ivec3(0,1,1));
            float f101 = getVoxelValue(voxelCoords + ivec3(1,0,1));
            float f110 = getVoxelValue(voxelCoords + ivec3(1,1,0));
            float f111 = getVoxelValue(voxelCoords + ivec3(1,1,1));

            float m000 = getPropagatedValue(voxelCoords + ivec3(0,0,0));
            float m100 = getPropagatedValue(voxelCoords + ivec3(1,0,0));
            float m010 = getPropagatedValue(voxelCoords + ivec3(0,1,0));
            float m001 = getPropagatedValue(voxelCoords + ivec3(0,0,1));
            float m011 = getPropagatedValue(voxelCoords + ivec3(0,1,1));
            float m101 = getPropagatedValue(voxelCoords + ivec3(1,0,1));
            float m110 = getPropagatedValue(voxelCoords + ivec3(1,1,0));
            float m111 = getPropagatedValue(voxelCoords + ivec3(1,1,1));

            float M000 = 0.0;
            float M010 = 0.0;
            float M100 = 0.0;
            float M110 = 0.0;

            M000 = max(M000, f001);
            M000 = max(M000, f011);
            M000 = max(M000, f101);
            M000 = max(M000, f111);
            M000 = max(M000, (f000 + f100 + f001) / 3.0);
            M000 = max(M000, (f000 + f010 + f001) / 3.0);
            M000 = max(M000, (f100 + f010 + f001) / 3.0);
            M000 = max(M000, (f100 + f001 + f101) / 3.0);
            M000 = max(M000, (f010 + f001 + f011) / 3.0);
            M000 = max(M000, (f110 + f101 + f011) / 3.0);

            M010 = max(M010, f011);
            M010 = max(M010, f111);
            M010 = max(M010, (f110 + f011 + f010) / 3.0);
            M010 = max(M010, (f110 + f011 + f111) / 3.0);

            M100 = max(M100, f101);
            M100 = max(M100, f111);
            M100 = max(M100, (f110 + f101 + f100) / 3.0);
            M100 = max(M100, (f110 + f101 + f111) / 3.0);

            M110 = max(M110, f111);

            bool occlusion =
                m000 >= M000 &&
                m001 >= M000 &&
                m010 >= M010 &&
                m011 >= M010 &&
                m100 >= M100 &&
                m101 >= M100 &&
                m110 >= M110;

            // bool occlusion = mmin(m000, m001, m010, m011, m100, m101, m110) >= mmax(M000, M010, M100, M110);

            setOutput(float(occlusion));
        }
        `
    }
}

class GPGPUFrontOcclusionMap implements GPGPUProgram    
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = false
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

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float mmax(float a1, float a2, float a3, float a4)
        {
            return max(max(max(a1, a2), a3), a4);
        }

        float mmin(float a1, float a2, float a3, float a4, float a5, float a6, float a7)
        {
            return min(min(min(min(min(min(a1, a2), a3), a4), a5), a6), a7);
        }

        ivec3 getCellCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getVoxelValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        float getPropagatedValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getB(safeCoords.z, safeCoords.y, safeCoords.x);
        }    

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            ivec3 voxelCoords = cellCoords - 1;
            
            float f000 = getVoxelValue(voxelCoords + ivec3(0,0,0));
            float f100 = getVoxelValue(voxelCoords + ivec3(1,0,0));
            float f010 = getVoxelValue(voxelCoords + ivec3(0,1,0));
            float f001 = getVoxelValue(voxelCoords + ivec3(0,0,1));
            float f011 = getVoxelValue(voxelCoords + ivec3(0,1,1));
            float f101 = getVoxelValue(voxelCoords + ivec3(1,0,1));
            float f110 = getVoxelValue(voxelCoords + ivec3(1,1,0));
            float f111 = getVoxelValue(voxelCoords + ivec3(1,1,1));

            float m111 = getPropagatedValue(voxelCoords + ivec3(1,1,1));
            float m110 = getPropagatedValue(voxelCoords + ivec3(1,1,0));
            float m101 = getPropagatedValue(voxelCoords + ivec3(1,0,1));
            float m011 = getPropagatedValue(voxelCoords + ivec3(0,1,1));
            float m001 = getPropagatedValue(voxelCoords + ivec3(0,0,1));
            float m010 = getPropagatedValue(voxelCoords + ivec3(0,1,0));
            float m100 = getPropagatedValue(voxelCoords + ivec3(1,0,0));
            float m000 = getPropagatedValue(voxelCoords + ivec3(0,0,0));

            float M111 = 0.0;
            float M101 = 0.0;
            float M011 = 0.0;
            float M001 = 0.0;

            M111 = max(M111, f110);
            M111 = max(M111, f100);
            M111 = max(M111, f010);
            M111 = max(M111, f000);
            M111 = max(M111, (f111 + f011 + f110) / 3.0);
            M111 = max(M111, (f111 + f101 + f110) / 3.0);
            M111 = max(M111, (f011 + f101 + f110) / 3.0);
            M111 = max(M111, (f011 + f110 + f010) / 3.0);
            M111 = max(M111, (f101 + f110 + f100) / 3.0);
            M111 = max(M111, (f001 + f010 + f100) / 3.0);

            M101 = max(M101, f100);
            M101 = max(M101, f000);
            M101 = max(M101, (f001 + f100 + f101) / 3.0);
            M101 = max(M101, (f001 + f100 + f000) / 3.0);

            M011 = max(M011, f010);
            M011 = max(M011, f000);
            M011 = max(M011, (f001 + f010 + f011) / 3.0);
            M011 = max(M011, (f001 + f010 + f000) / 3.0);

            M001 = max(M001, f000);

            bool occlusion =
                m111 >= M111 &&
                m110 >= M111 &&
                m101 >= M101 &&
                m100 >= M101 &&
                m011 >= M011 &&
                m010 >= M011 &&
                m001 >= M001;

            // bool occlusion = mmin(m111, m110, m101, m100, m011, m010, m001) >= max(M111, M101, M011, M001);

            setOutput(float(occlusion));
        }
        `
    }
}

class GPGPUBackOcclusionMap2 implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = false
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

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});
   
        ivec3 getCellCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getVoxelValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        float getPropagatedValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getB(safeCoords.z, safeCoords.y, safeCoords.x);
        }    

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            ivec3 voxelCoords = cellCoords - 1;
            
            float f000 = getVoxelValue(voxelCoords + ivec3(0,0,0));
            float f100 = getVoxelValue(voxelCoords + ivec3(1,0,0));
            float f010 = getVoxelValue(voxelCoords + ivec3(0,1,0));
            float f001 = getVoxelValue(voxelCoords + ivec3(0,0,1));
            float f011 = getVoxelValue(voxelCoords + ivec3(0,1,1));
            float f101 = getVoxelValue(voxelCoords + ivec3(1,0,1));
            float f110 = getVoxelValue(voxelCoords + ivec3(1,1,0));
            float f111 = getVoxelValue(voxelCoords + ivec3(1,1,1));

            float m000 = getPropagatedValue(voxelCoords + ivec3(0,0,0));
            float m100 = getPropagatedValue(voxelCoords + ivec3(1,0,0));
            float m010 = getPropagatedValue(voxelCoords + ivec3(0,1,0));
            float m001 = getPropagatedValue(voxelCoords + ivec3(0,0,1));
            float m011 = getPropagatedValue(voxelCoords + ivec3(0,1,1));
            float m101 = getPropagatedValue(voxelCoords + ivec3(1,0,1));
            float m110 = getPropagatedValue(voxelCoords + ivec3(1,1,0));
            float m111 = getPropagatedValue(voxelCoords + ivec3(1,1,1));

            float M000 = 0.0;
            float M010 = 0.0;
            float M100 = 0.0;
            float M110 = 0.0;

            M000 = max(M000, f001);
            M000 = max(M000, f011);
            M000 = max(M000, f101);
            M000 = max(M000, f111);
            M000 = max(M000, (f000 + f100 + f001) / 3.0);
            M000 = max(M000, (f000 + f010 + f001) / 3.0);
            M000 = max(M000, (f100 + f010 + f001) / 3.0);
            M000 = max(M000, (f100 + f001 + f101) / 3.0);
            M000 = max(M000, (f010 + f001 + f011) / 3.0);
            M000 = max(M000, (f110 + f101 + f011) / 3.0);

            M010 = max(M010, f011);
            M010 = max(M010, f111);
            M010 = max(M010, (f110 + f011 + f010) / 3.0);
            M010 = max(M010, (f110 + f011 + f111) / 3.0);

            M100 = max(M100, f101);
            M100 = max(M100, f111);
            M100 = max(M100, (f110 + f101 + f100) / 3.0);
            M100 = max(M100, (f110 + f101 + f111) / 3.0);

            M110 = max(M110, f111);

            bool occlusion =
                m000 >= M000 && m001 != f001 &&
                m010 >= M010 && m011 != f011 &&
                m100 >= M100 && m101 != f101 &&
                m110 >= M110;

            setOutput(float(occlusion));
        }
        `
    }
}

class GPGPUFrontOcclusionMap2 implements GPGPUProgram    
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = false
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

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        ivec3 getCellCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getVoxelValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        float getPropagatedValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getB(safeCoords.z, safeCoords.y, safeCoords.x);
        }    

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            ivec3 voxelCoords = cellCoords - 1;
            
            float f000 = getVoxelValue(voxelCoords + ivec3(0,0,0));
            float f100 = getVoxelValue(voxelCoords + ivec3(1,0,0));
            float f010 = getVoxelValue(voxelCoords + ivec3(0,1,0));
            float f001 = getVoxelValue(voxelCoords + ivec3(0,0,1));
            float f011 = getVoxelValue(voxelCoords + ivec3(0,1,1));
            float f101 = getVoxelValue(voxelCoords + ivec3(1,0,1));
            float f110 = getVoxelValue(voxelCoords + ivec3(1,1,0));
            float f111 = getVoxelValue(voxelCoords + ivec3(1,1,1));

            float m111 = getPropagatedValue(voxelCoords + ivec3(1,1,1));
            float m110 = getPropagatedValue(voxelCoords + ivec3(1,1,0));
            float m101 = getPropagatedValue(voxelCoords + ivec3(1,0,1));
            float m011 = getPropagatedValue(voxelCoords + ivec3(0,1,1));
            float m001 = getPropagatedValue(voxelCoords + ivec3(0,0,1));
            float m010 = getPropagatedValue(voxelCoords + ivec3(0,1,0));
            float m100 = getPropagatedValue(voxelCoords + ivec3(1,0,0));
            float m000 = getPropagatedValue(voxelCoords + ivec3(0,0,0));

            float M111 = 0.0;
            float M101 = 0.0;
            float M011 = 0.0;
            float M001 = 0.0;

            M111 = max(M111, f110);
            M111 = max(M111, f100);
            M111 = max(M111, f010);
            M111 = max(M111, f000);
            M111 = max(M111, (f111 + f011 + f110) / 3.0);
            M111 = max(M111, (f111 + f101 + f110) / 3.0);
            M111 = max(M111, (f011 + f101 + f110) / 3.0);
            M111 = max(M111, (f011 + f110 + f010) / 3.0);
            M111 = max(M111, (f101 + f110 + f100) / 3.0);
            M111 = max(M111, (f001 + f010 + f100) / 3.0);

            M101 = max(M101, f100);
            M101 = max(M101, f000);
            M101 = max(M101, (f001 + f100 + f101) / 3.0);
            M101 = max(M101, (f001 + f100 + f000) / 3.0);

            M011 = max(M011, f010);
            M011 = max(M011, f000);
            M011 = max(M011, (f001 + f010 + f011) / 3.0);
            M011 = max(M011, (f001 + f010 + f000) / 3.0);

            M001 = max(M001, f000);

            bool occlusion =
                m111 >= M111 && m110 != f110 &&
                m101 >= M101 && m100 != f100 &&
                m011 >= M011 && m010 != f010 &&
                m001 >= M001;

            setOutput(float(occlusion));
        }
        `
    }
}

class GPGPUBackOcclusionMap3 implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = false
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

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});
   
        float mmin(float a, float b, float c)
        {
            return min(min(a, b), c);
        }

        float mmax(float a, float b, float c)
        {
            return max(max(a, b), c);
        }

        ivec3 getCellCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getVoxelValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        float getPropagatedValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getB(safeCoords.z, safeCoords.y, safeCoords.x);
        }    

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            ivec3 voxelCoords = cellCoords - 1;
            
            float f000 = getVoxelValue(voxelCoords + ivec3(0,0,0));
            float fP00 = getVoxelValue(voxelCoords + ivec3(1,0,0));
            float f0P0 = getVoxelValue(voxelCoords + ivec3(0,1,0));
            float f00P = getVoxelValue(voxelCoords + ivec3(0,0,1));
            float f0PP = getVoxelValue(voxelCoords + ivec3(0,1,1));
            float fP0P = getVoxelValue(voxelCoords + ivec3(1,0,1));
            float fPP0 = getVoxelValue(voxelCoords + ivec3(1,1,0));
            float fPPP = getVoxelValue(voxelCoords + ivec3(1,1,1));

            float mNNN = getPropagatedValue(voxelCoords + ivec3(-1,-1,-1));
            float mNN0 = getPropagatedValue(voxelCoords + ivec3(-1,-1, 0));
            float mN0N = getPropagatedValue(voxelCoords + ivec3(-1, 0,-1));
            float mN00 = getPropagatedValue(voxelCoords + ivec3(-1, 0, 0));
            float mNPN = getPropagatedValue(voxelCoords + ivec3(-1, 1,-1));
            float mNP0 = getPropagatedValue(voxelCoords + ivec3(-1, 1, 0));
            float m0NN = getPropagatedValue(voxelCoords + ivec3( 0,-1,-1));
            float m0N0 = getPropagatedValue(voxelCoords + ivec3( 0,-1, 0));
            float m00N = getPropagatedValue(voxelCoords + ivec3( 0, 0,-1));
            float m000 = getPropagatedValue(voxelCoords + ivec3( 0, 0, 0));
            float m0PN = getPropagatedValue(voxelCoords + ivec3( 0, 1,-1));
            float m0P0 = getPropagatedValue(voxelCoords + ivec3( 0, 1, 0));
            float mPNN = getPropagatedValue(voxelCoords + ivec3( 1,-1,-1));
            float mPN0 = getPropagatedValue(voxelCoords + ivec3( 1,-1, 0));
            float mP0N = getPropagatedValue(voxelCoords + ivec3( 1, 0,-1));
            float mP00 = getPropagatedValue(voxelCoords + ivec3( 1, 0, 0));
            float mPPN = getPropagatedValue(voxelCoords + ivec3( 1, 1,-1));
            float mPP0 = getPropagatedValue(voxelCoords + ivec3( 1, 1, 0));

            float mNN0_N00_0N0_000 = min(min(min(mNN0, mN00), m0N0), m000);
            float mN00_NP0_000_0P0 = min(min(min(mN00, mNP0), m000), m0P0);
            float m0N0_000_PN0_P00 = min(min(min(m0N0, m000), mPN0), mP00);
            float m000_P00_0P0_PP0 = min(min(min(m000, mP00), m0P0), mPP0);

            float a00P_000_0P0 = (f00P + f000 + f0P0) / 3.0;
            float a00P_0PP_0P0 = (f00P + f0PP + f0P0) / 3.0;
            float a00P_000_P00 = (f00P + f000 + fP00) / 3.0;
            float a00P_P0P_P00 = (f00P + fP0P + fP00) / 3.0;
            float aP00_0P0_00P = (fP00 + f0P0 + f00P) / 3.0;
            float a0PP_P0P_PP0 = (f0PP + fP0P + fPP0) / 3.0;
            float a0PP_0P0_PP0 = (f0PP + f0P0 + fPP0) / 3.0;
            float a0PP_PPP_PP0 = (f0PP + fPPP + fPP0) / 3.0;
            float aP0P_P00_PP0 = (fP0P + fP00 + fPP0) / 3.0;
            float aP0P_PPP_PP0 = (fP0P + fPPP + fPP0) / 3.0;

            float t0NN = max(f0PP, max(a00P_000_0P0, a00P_0PP_0P0));
            float tN0N = max(f0PP, max(a00P_000_P00, a00P_P0P_P00));
            float tNNN = max(fPPP, max(aP00_0P0_00P, a0PP_P0P_PP0));
            float tNPN = max(fPPP, max(a0PP_0P0_PP0, a0PP_PPP_PP0));
            float tPNN = max(fPPP, max(aP0P_P00_PP0, aP0P_PPP_PP0));

            bool occlusion = true

            && max(mN00, f000) >= fP00
            && max(m0N0, f000) >= f0P0
            && max(m00N, f000) >= f00P
            && max(m0NN, f000) >= t0NN
            && max(mN0N, f000) >= tN0N
            && max(mNNN, f000) >= tNNN

            && min(m00N, mN0N) >= f0P0
            && max(m0PN, f0P0) >= f0PP
            && max(mNPN, f0P0) >= tNPN

            && min(m00N, m0NN) >= fP00
            && max(mP0N, fP00) >= fP0P
            && max(mPNN, fP00) >= tPNN

            && mmin(m0PN, m00N, mP0N) >= fPP0
            && max(mPPN, fPP0) >= fPPP

            && mNN0_N00_0N0_000 >= f00P
            && mN00_NP0_000_0P0 >= f0PP
            && m0N0_000_PN0_P00 >= fP0P
            && m000_P00_0P0_PP0 >= fPPP;

            setOutput(float(occlusion));
        }
        `
    }
}

class GPGPUFrontOcclusionMap3 implements GPGPUProgram 
{
  variableNames = ['A', 'B']
  outputShape: number[]
  userCode: string
  packedInputs = false
  packedOutput = false

  constructor(inputShape: [number, number, number]) 
  {
    const [inDepth, inHeight, inWidth] = inputShape
    const [outDepth, outHeight, outWidth] = [inDepth, inHeight, inWidth].map((x: number) => x + 1)
    this.outputShape = [outDepth, outHeight, outWidth]

    this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});
   
        float mmin(float a, float b, float c)
        {
            return min(min(a, b), c);
        }

        float mmax(float a, float b, float c)
        {
            return max(max(a, b), c);
        }

        ivec3 getCellCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getVoxelValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        float getPropagatedValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getB(safeCoords.z, safeCoords.y, safeCoords.x);
        }    

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            ivec3 voxelCoords = cellCoords - 1;

            float f111 = getVoxelValue(voxelCoords + ivec3(1,1,1));
            float f011 = getVoxelValue(voxelCoords + ivec3(0,1,1));
            float f101 = getVoxelValue(voxelCoords + ivec3(1,0,1));
            float f110 = getVoxelValue(voxelCoords + ivec3(1,1,0));
            float f100 = getVoxelValue(voxelCoords + ivec3(1,0,0));
            float f010 = getVoxelValue(voxelCoords + ivec3(0,1,0));
            float f001 = getVoxelValue(voxelCoords + ivec3(0,0,1));
            float f000 = getVoxelValue(voxelCoords + ivec3(0,0,0));

            float m222 = getPropagatedValue(voxelCoords + ivec3(2,2,2));
            float m221 = getPropagatedValue(voxelCoords + ivec3(2,2,1));
            float m212 = getPropagatedValue(voxelCoords + ivec3(2,1,2));
            float m211 = getPropagatedValue(voxelCoords + ivec3(2,1,1));
            float m202 = getPropagatedValue(voxelCoords + ivec3(2,0,2));
            float m201 = getPropagatedValue(voxelCoords + ivec3(2,0,1));
            float m122 = getPropagatedValue(voxelCoords + ivec3(1,2,2));
            float m121 = getPropagatedValue(voxelCoords + ivec3(1,2,1));
            float m112 = getPropagatedValue(voxelCoords + ivec3(1,1,2));
            float m102 = getPropagatedValue(voxelCoords + ivec3(1,0,2));
            float m022 = getPropagatedValue(voxelCoords + ivec3(0,2,2));
            float m021 = getPropagatedValue(voxelCoords + ivec3(0,2,1));
            float m012 = getPropagatedValue(voxelCoords + ivec3(0,1,2));
            float m002 = getPropagatedValue(voxelCoords + ivec3(0,0,2));

            bool occlusion = true

            && max(m211, f111) >= f011
            && max(m121, f111) >= f101
            && max(m112, f111) >= f110
            && max(m122, f111) >= mmax(f100, (f110 + f111 + f101)/3.0, (f110 + f100 + f101)/3.0)
            && max(m212, f111) >= mmax(f100, (f110 + f111 + f011)/3.0, (f110 + f010 + f011)/3.0)
            && max(m222, f111) >= mmax(f000, (f011 + f101 + f110)/3.0, (f100 + f010 + f001)/3.0)

            && min(m112, m212) >= f101
            && max(m102, f101) >= f100
            && max(m202, f101) >= mmax(f000, (f100 + f101 + f001)/3.0, (f100 + f000 + f001)/3.0)

            && min(m112, m122) >= f011
            && max(m012, f011) >= f010
            && max(m022, f011) >= mmax(f000, (f010 + f011 + f001)/3.0, (f010 + f000 + f001)/3.0)

            && mmin(m102, m112, m012) >= f001
            && max(m002, f001) >= f000

            && mmin(m211, m221, m121) >= f110

            && min(m201, m211) >= f100

            && min(m121, m021) >= f010;

            setOutput(float(occlusion));
        }
        `
  }
}



function runProgram(prog: GPGPUProgram, inputs: tf.Tensor[], outputDtype?: tf.DataType): tf.Tensor
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(prog, inputs, outputDtype)
    return tf.engine().makeTensorFromTensorInfo(info) as tf.Tensor
}

async function computeBackOcclusionMap(volumeMap: tf.Tensor3D) : Promise<tf.Tensor<tf.Rank>>
{
    const propagationProgram = new GPGPUBackPropagationMap(volumeMap.shape)
    const occlusionProgram = new GPGPUBackOcclusionMap3(volumeMap.shape)

    let propagationMap = runProgram(propagationProgram, [volumeMap])
    let maxPropagation = Math.max(...volumeMap.shape) / 2

    for (let i = 0; i < maxPropagation; i++) 
    {
        const prev = propagationMap
        propagationMap = runProgram(propagationProgram, [prev])
        prev.dispose()

        if (i%2==0) await tf.nextFrame()                     
    }

    const occlusionMap = runProgram(occlusionProgram, [volumeMap, propagationMap], 'bool')
    propagationMap.dispose()

    return occlusionMap as tf.Tensor
}

async function computeFrontOcclusionMap(volumeMap: tf.Tensor3D) : Promise<tf.Tensor<tf.Rank>>
{
    const propagationProgram = new GPGPUFrontPropagationMap(volumeMap.shape)
    const occlusionProgram = new GPGPUFrontOcclusionMap3(volumeMap.shape)

    let propagationMap = runProgram(propagationProgram, [volumeMap])
    let maxPropagation = Math.max(...volumeMap.shape) / 2

    for (let i = 0; i < maxPropagation; i++) 
    {
        const prev = propagationMap
        propagationMap = runProgram(propagationProgram, [prev])
        prev.dispose()

        if (i%2==0) await tf.nextFrame()                     
    }

    const occlusionMap = runProgram(occlusionProgram, [volumeMap, propagationMap], 'bool')
    propagationMap.dispose()

    return occlusionMap as tf.Tensor
}

export async function computeOcclusionMap(volumeMap: tf.Tensor3D) : Promise<tf.Tensor<tf.Rank>>
{
    const backOcclusionMap = await computeBackOcclusionMap(volumeMap)
    const frontOcclusionMap = await computeFrontOcclusionMap(volumeMap)

    const occlusionMap = tf.logicalOr(backOcclusionMap, frontOcclusionMap);
    tf.dispose([backOcclusionMap, frontOcclusionMap]);

    return occlusionMap as tf.Tensor
}

