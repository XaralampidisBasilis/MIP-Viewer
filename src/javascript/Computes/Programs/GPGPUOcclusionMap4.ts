import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'


// class GPGPUBackPropagationMap implements GPGPUProgram 
// {
//     variableNames = ['A', 'B']
//     outputShape: number[]
//     userCode: string
//     packedInputs = true
//     packedOutput = true

//     constructor
//     (
//         inputShape: [number, number, number], 
//     ) 
//     {
//         const [inDepth, inHeight, inWidth] = inputShape
//         this.outputShape = [inDepth, inHeight, inWidth]  
//         this.userCode = `

//         const ivec3 minCoords = ivec3(0);
//         const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

//         float mmin(float a, float b, float c, float d)
//         {
//             return min(min(min(a, b), c), d);
//         }

//         ivec3 getVoxelCoords()
//         {
//             ivec3 outCoords = getOutputCoords();
//             return ivec3(outCoords.z, outCoords.y, outCoords.x);
//         }

//         vec4 getPackedValues(ivec3 voxelCoords)
//         {
//             ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);

//             bool insideWidth  = safeCoords.x < ${inWidth-1};
//             bool insideHeight = safeCoords.y < ${inHeight-1};

//             vec4 packedValues = getA(safeCoords.z, safeCoords.y, safeCoords.x);
//             packedValues.ga = insideWidth ? packedValues.ga : packedValues.rb;
//             packedValues.ba = insideHeight ? packedValues.ba : packedValues.rg;

//             return packedValues;
//         }

//         void main()
//         {
//             ivec3 voxelCoords = getVoxelCoords();

//             vec4 f000 = getPackedValues(voxelCoords - ivec3(0,0,0));
//             vec4 f001 = getPackedValues(voxelCoords - ivec3(0,0,1));
//             vec4 v201 = getPackedValues(voxelCoords - ivec3(2,0,1));
//             vec4 v021 = getPackedValues(voxelCoords - ivec3(0,2,1));
//             vec4 v221 = getPackedValues(voxelCoords - ivec3(2,2,1));

//             f000.r = max(f000.r, mmin(f001.r, v201.g, v021.b, v221.a));
//             f000.g = max(f000.g, mmin(f001.r, f001.g, v021.b, v021.a));
//             f000.b = max(f000.b, mmin(f001.r, v201.g, f001.b, v201.a));
//             f000.a = max(f000.a, mmin(f001.r, f001.g, f001.b, f001.a));

//             setOutput(f000);
//         }
//         `
//     }
// }

// class GPGPUFrontPropagationMap implements GPGPUProgram 
// {
//     variableNames = ['A']
//     outputShape: number[]
//     userCode: string
//     packedInputs = true
//     packedOutput = true

//     constructor
//     (
//         inputShape: [number, number, number], 
//     ) 
//     {
//         const [inDepth, inHeight, inWidth] = inputShape
//         this.outputShape = [inDepth, inHeight, inWidth]  
//         this.userCode = `

//         const ivec3 minCoords = ivec3(0);
//         const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

//         float mmin(float a, float b, float c, float d)
//         {
//             return min(min(min(a, b), c), d);
//         }

//         ivec3 getVoxelCoords()
//         {
//             ivec3 outCoords = getOutputCoords();
//             return ivec3(outCoords.z, outCoords.y, outCoords.x);
//         }

//         vec4 getPackedValues(ivec3 voxelCoords)
//         {
//             ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);

//             bool insideWidth  = safeCoords.x < ${inWidth-1};
//             bool insideHeight = safeCoords.y < ${inHeight-1};

//             vec4 packedValues = getA(safeCoords.z, safeCoords.y, safeCoords.x);
//             packedValues.ga = insideWidth ? packedValues.ga : packedValues.rb;
//             packedValues.ba = insideHeight ? packedValues.ba : packedValues.rg;

//             return packedValues;
//         }

//         void main()
//         {
//             ivec3 voxelCoords = getVoxelCoords();

//             vec4 f000 = getPackedValues(voxelCoords + ivec3(0,0,0));
//             vec4 f001 = getPackedValues(voxelCoords + ivec3(0,0,1));
//             vec4 v201 = getPackedValues(voxelCoords + ivec3(2,0,1));
//             vec4 v021 = getPackedValues(voxelCoords + ivec3(0,2,1));
//             vec4 v221 = getPackedValues(voxelCoords + ivec3(2,2,1));

//             f000.r = max(f000.r, mmin(f001.r, f001.g, f001.b, f001.a));
//             f000.g = max(f000.g, mmin(v201.r, f001.g, v201.b, f001.a));
//             f000.b = max(f000.b, mmin(v021.r, v021.g, f001.b, f001.a));
//             f000.a = max(f000.a, mmin(v221.r, v021.g, v201.b, f001.a));

//             setOutput(f000);
//         }
//         `
//     }
// }

class GPGPUStartPropagationMap implements GPGPUProgram 
{
    variableNames = ['A']
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

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
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

        void main()
        {
            ivec3 cellCoords = getCellCoords();

            float f000 = getVoxelValue(cellCoords - ivec3(0,0,0));
            float fN00 = getVoxelValue(cellCoords - ivec3(1,0,0));
            float f0N0 = getVoxelValue(cellCoords - ivec3(0,1,0));
            float fNN0 = getVoxelValue(cellCoords - ivec3(1,1,0));

            float F000 = min4(f000, fN00, f0N0, fNN0);

            setOutput(F000);
        }
        `
    }
}

class GPGPUBackPropagationMap implements GPGPUProgram 
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
        this.outputShape = [inDepth, inHeight, inWidth]  
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth}, ${inHeight}, ${inDepth});

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        ivec3 getCellCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getInitialCellValue(ivec3 cellCoords)
        {
            ivec3 safeCoords = clamp(cellCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        float getPropagateCellValue(ivec3 cellCoords)
        {
            ivec3 safeCoords = clamp(cellCoords, minCoords, maxCoords);
            return getB(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        void main()
        {
            ivec3 cellCoords = getCellCoords();

            float F000 = getInitialCellValue(cellCoords);

            float L00N = getPropagateCellValue(cellCoords - ivec3(0,0,1));
            float LN0N = getPropagateCellValue(cellCoords - ivec3(1,0,1));
            float L0NN = getPropagateCellValue(cellCoords - ivec3(0,1,1));
            float LNNN = getPropagateCellValue(cellCoords - ivec3(1,1,1));

            F000 = max(F000, min4(L00N, LN0N, L0NN, LNNN));

            setOutput(F000);
        }
        `
    }
}

class GPGPUFrontPropagationMap implements GPGPUProgram 
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
        this.outputShape = [inDepth, inHeight, inWidth]  
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth}, ${inHeight}, ${inDepth});

        float min4(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        ivec3 getCellCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getInitialCellValue(ivec3 cellCoords)
        {
            ivec3 safeCoords = clamp(cellCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        float getPropagateCellValue(ivec3 cellCoords)
        {
            ivec3 safeCoords = clamp(cellCoords, minCoords, maxCoords);
            return getB(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        void main()
        {
            ivec3 cellCoords = getCellCoords();

            float F000 = getInitialCellValue(cellCoords);

            float L00P = getPropagateCellValue(cellCoords + ivec3(0,0,1));
            float LP0P = getPropagateCellValue(cellCoords + ivec3(1,0,1));
            float L0PP = getPropagateCellValue(cellCoords + ivec3(0,1,1));
            float LPPP = getPropagateCellValue(cellCoords + ivec3(1,1,1));

            F000 = max(F000, min4(L00P, LP0P, L0PP, LPPP));
            
            setOutput(F000);
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
    
        float avg3(float a, float b, float c) { return (a + b + c) / 3.0; }
        float min3(float a, float b, float c) { return min(min(a, b), c); }
        float max3(float a, float b, float c) { return max(max(a, b), c); }
        
        ivec3 getCellCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getMinVertexValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getB(safeCoords.z, safeCoords.y, safeCoords.x);
        }    

        float getVertexValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            ivec3 voxelCoords = cellCoords - 1;

            float mNNN = getMinVertexValue(voxelCoords + ivec3(-1,-1,-1));
            float mNN0 = getMinVertexValue(voxelCoords + ivec3(-1,-1, 0));
            float mN0N = getMinVertexValue(voxelCoords + ivec3(-1, 0,-1));
            float mN00 = getMinVertexValue(voxelCoords + ivec3(-1, 0, 0));
            float mNPN = getMinVertexValue(voxelCoords + ivec3(-1, 1,-1));
            float mNP0 = getMinVertexValue(voxelCoords + ivec3(-1, 1, 0));
            float m0NN = getMinVertexValue(voxelCoords + ivec3( 0,-1,-1));
            float m0N0 = getMinVertexValue(voxelCoords + ivec3( 0,-1, 0));
            float m00N = getMinVertexValue(voxelCoords + ivec3( 0, 0,-1));
            float m0PN = getMinVertexValue(voxelCoords + ivec3( 0, 1,-1));
            float mPNN = getMinVertexValue(voxelCoords + ivec3( 1,-1,-1));
            float mPN0 = getMinVertexValue(voxelCoords + ivec3( 1,-1, 0));
            float mP0N = getMinVertexValue(voxelCoords + ivec3( 1, 0,-1));
            float mPPN = getMinVertexValue(voxelCoords + ivec3( 1, 1,-1));
            
            float f000 = getVertexValue(voxelCoords + ivec3(0,0,0));
            float fP00 = getVertexValue(voxelCoords + ivec3(1,0,0));
            float f0P0 = getVertexValue(voxelCoords + ivec3(0,1,0));
            float f00P = getVertexValue(voxelCoords + ivec3(0,0,1));
            float f0PP = getVertexValue(voxelCoords + ivec3(0,1,1));
            float fP0P = getVertexValue(voxelCoords + ivec3(1,0,1));
            float fPP0 = getVertexValue(voxelCoords + ivec3(1,1,0));
            float fPPP = getVertexValue(voxelCoords + ivec3(1,1,1));

            float a00P_000_0P0 = avg3(f00P, f000, f0P0);
            float a00P_0PP_0P0 = avg3(f00P, f0PP, f0P0);
            float a00P_000_P00 = avg3(f00P, f000, fP00);
            float a00P_P0P_P00 = avg3(f00P, fP0P, fP00);
            float aP00_0P0_00P = avg3(fP00, f0P0, f00P);
            float a0PP_P0P_PP0 = avg3(f0PP, fP0P, fPP0);
            float a0PP_0P0_PP0 = avg3(f0PP, f0P0, fPP0);
            float a0PP_PPP_PP0 = avg3(f0PP, fPPP, fPP0);
            float aP0P_P00_PP0 = avg3(fP0P, fP00, fPP0);
            float aP0P_PPP_PP0 = avg3(fP0P, fPPP, fPP0);

            float t000_00P_0P0_0PP = max3(a00P_000_0P0, a00P_0PP_0P0, f0PP);
            float t000_00P_P00_P0P = max3(a00P_000_P00, a00P_P0P_P00, f0PP);
            float t000_000_PPP_PPP = max3(aP00_0P0_00P, a0PP_P0P_PP0, fPPP);
            float t0P0_0PP_PP0_PPP = max3(a0PP_0P0_PP0, a0PP_PPP_PP0, fPPP);
            float tP00_P0P_PP0_PPP = max3(aP0P_P00_PP0, aP0P_PPP_PP0, fPPP);

            bool occ0 = 
            f000 >= t000_00P_0P0_0PP && 
            f000 >= t000_00P_P00_P0P && 
            f000 >= t000_000_PPP_PPP && 
            f0P0 >= t0P0_0PP_PP0_PPP && 
            fP00 >= tP00_P0P_PP0_PPP && 
            fPP0 >= fPPP;

            bool occ1 =
            mNNN >= max(f000, t000_000_PPP_PPP) &&
            m0NN >= max(f000, t000_00P_0P0_0PP) &&
            mN0N >= max(f000, t000_00P_P00_P0P) &&
            mNPN >= max(f0P0, t0P0_0PP_PP0_PPP) && 
            mPNN >= max(fP00, tP00_P0P_PP0_PPP) &&
            m00N >= max(f000, f00P) &&
            m0PN >= max(f0P0, f0PP) &&
            mP0N >= max(fP00, fP0P) &&
            mPPN >= max(fPP0, fPPP) &&
            mP0N >= fPP0 &&
            m0PN >= fPP0 &&
            m00N >= fPP0 &&
            mNN0 >= f00P &&
            mN00 >= f00P &&
            m0N0 >= f00P &&
            m00N >= f0P0 &&
            mN0N >= f0P0 &&
            m00N >= fP00 &&
            m0NN >= fP00 &&
            mNP0 >= f0PP && 
            mN00 >= f0PP && 
            mPN0 >= fP0P && 
            m0N0 >= fP0P;

            setOutput(float(occ0 || occ1));
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
    
        float avg3(float a, float b, float c) { return (a + b + c) / 3.0; }
        float max3(float a, float b, float c) { return max(max(a, b), c); }
        
        ivec3 getCellCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getMinVertexValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getB(safeCoords.z, safeCoords.y, safeCoords.x);
        }    

        float getVertexValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            ivec3 voxelCoords = cellCoords - 1;

            // Applied transformation (x,y,z) -> (1-x, 1-y, 1-z)
            float mNNN = getMinVertexValue(voxelCoords + ivec3(2,2,2));
            float mNN0 = getMinVertexValue(voxelCoords + ivec3(2,2,1));
            float mN0N = getMinVertexValue(voxelCoords + ivec3(2,1,2));
            float mN00 = getMinVertexValue(voxelCoords + ivec3(2,1,1));
            float mNPN = getMinVertexValue(voxelCoords + ivec3(2,0,2));
            float mNP0 = getMinVertexValue(voxelCoords + ivec3(2,0,1));
            float m0NN = getMinVertexValue(voxelCoords + ivec3(1,2,2));
            float m0N0 = getMinVertexValue(voxelCoords + ivec3(1,2,1));
            float m00N = getMinVertexValue(voxelCoords + ivec3(1,1,2));
            float m0PN = getMinVertexValue(voxelCoords + ivec3(1,0,2));
            float mPNN = getMinVertexValue(voxelCoords + ivec3(0,2,2));
            float mPN0 = getMinVertexValue(voxelCoords + ivec3(0,2,1));
            float mP0N = getMinVertexValue(voxelCoords + ivec3(0,1,2));
            float mPPN = getMinVertexValue(voxelCoords + ivec3(0,0,2));
            
            // Applied transformation (x,y,z) -> (1-x, 1-y, 1-z)
            float f000 = getVertexValue(voxelCoords + ivec3(1,1,1));
            float fP00 = getVertexValue(voxelCoords + ivec3(0,1,1));
            float f0P0 = getVertexValue(voxelCoords + ivec3(1,0,1));
            float f00P = getVertexValue(voxelCoords + ivec3(1,1,0));
            float f0PP = getVertexValue(voxelCoords + ivec3(1,0,0));
            float fP0P = getVertexValue(voxelCoords + ivec3(0,1,0));
            float fPP0 = getVertexValue(voxelCoords + ivec3(0,0,1));
            float fPPP = getVertexValue(voxelCoords + ivec3(0,0,0));

            float a00P_000_0P0 = avg3(f00P, f000, f0P0);
            float a00P_0PP_0P0 = avg3(f00P, f0PP, f0P0);
            float a00P_000_P00 = avg3(f00P, f000, fP00);
            float a00P_P0P_P00 = avg3(f00P, fP0P, fP00);
            float aP00_0P0_00P = avg3(fP00, f0P0, f00P);
            float a0PP_P0P_PP0 = avg3(f0PP, fP0P, fPP0);
            float a0PP_0P0_PP0 = avg3(f0PP, f0P0, fPP0);
            float a0PP_PPP_PP0 = avg3(f0PP, fPPP, fPP0);
            float aP0P_P00_PP0 = avg3(fP0P, fP00, fPP0);
            float aP0P_PPP_PP0 = avg3(fP0P, fPPP, fPP0);

            float t000_00P_0P0_0PP = max3(a00P_000_0P0, a00P_0PP_0P0, f0PP);
            float t000_00P_P00_P0P = max3(a00P_000_P00, a00P_P0P_P00, f0PP);
            float t0P0_0PP_PP0_PPP = max3(a0PP_0P0_PP0, a0PP_PPP_PP0, fPPP);
            float tP00_P0P_PP0_PPP = max3(aP0P_P00_PP0, aP0P_PPP_PP0, fPPP);
            float t000_000_PPP_PPP = max3(aP00_0P0_00P, a0PP_P0P_PP0, fPPP);

            bool occ0 = 
            f000 >= t000_00P_0P0_0PP && 
            f000 >= t000_00P_P00_P0P && 
            f000 >= t000_000_PPP_PPP && 
            f0P0 >= t0P0_0PP_PP0_PPP && 
            fP00 >= tP00_P0P_PP0_PPP && 
            fPP0 >= fPPP;

            bool occ1 =
            mNNN >= max(f000, t000_000_PPP_PPP) &&
            m0NN >= max(f000, t000_00P_0P0_0PP) &&
            mN0N >= max(f000, t000_00P_P00_P0P) &&
            mNPN >= max(f0P0, t0P0_0PP_PP0_PPP) && 
            mPNN >= max(fP00, tP00_P0P_PP0_PPP) &&
            m00N >= max(f000, f00P) &&
            m0PN >= max(f0P0, f0PP) &&
            mP0N >= max(fP00, fP0P) &&
            mPPN >= max(fPP0, fPPP) &&
            m00N >= f0P0 &&
            mN0N >= f0P0 &&
            m00N >= fP00 &&
            m0NN >= fP00 &&
            mP0N >= fPP0 &&
            m0PN >= fPP0 &&
            m00N >= fPP0 &&
            mNN0 >= f00P &&
            mN00 >= f00P &&
            m0N0 >= f00P &&
            mNP0 >= f0PP && 
            mN00 >= f0PP && 
            mPN0 >= fP0P && 
            m0N0 >= fP0P;

            setOutput(float(occ0 || occ1));
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
    
        float avg3(float a, float b, float c) { return (a + b + c) / 3.0; }
        float min3(float a, float b, float c) { return min(min(a, b), c); }
        float max3(float a, float b, float c) { return max(max(a, b), c); }
        
        ivec3 getCellCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getMinVertexValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getB(safeCoords.z, safeCoords.y, safeCoords.x);
        }    

        float getVertexValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            ivec3 voxelCoords = cellCoords - 1;

            float mNNN = getMinVertexValue(voxelCoords + ivec3(-1,-1,-1));
            float mNN0 = getMinVertexValue(voxelCoords + ivec3(-1,-1, 0));
            float mN0N = getMinVertexValue(voxelCoords + ivec3(-1, 0,-1));
            float mN00 = getMinVertexValue(voxelCoords + ivec3(-1, 0, 0));
            float mNPN = getMinVertexValue(voxelCoords + ivec3(-1, 1,-1));
            float mNP0 = getMinVertexValue(voxelCoords + ivec3(-1, 1, 0));
            float m0NN = getMinVertexValue(voxelCoords + ivec3( 0,-1,-1));
            float m0N0 = getMinVertexValue(voxelCoords + ivec3( 0,-1, 0));
            float m00N = getMinVertexValue(voxelCoords + ivec3( 0, 0,-1));
            float m0PN = getMinVertexValue(voxelCoords + ivec3( 0, 1,-1));
            float mPNN = getMinVertexValue(voxelCoords + ivec3( 1,-1,-1));
            float mPN0 = getMinVertexValue(voxelCoords + ivec3( 1,-1, 0));
            float mP0N = getMinVertexValue(voxelCoords + ivec3( 1, 0,-1));
            float mPPN = getMinVertexValue(voxelCoords + ivec3( 1, 1,-1));
            
            float f000 = getVertexValue(voxelCoords + ivec3(0,0,0));
            float fP00 = getVertexValue(voxelCoords + ivec3(1,0,0));
            float f0P0 = getVertexValue(voxelCoords + ivec3(0,1,0));
            float f00P = getVertexValue(voxelCoords + ivec3(0,0,1));
            float f0PP = getVertexValue(voxelCoords + ivec3(0,1,1));
            float fP0P = getVertexValue(voxelCoords + ivec3(1,0,1));
            float fPP0 = getVertexValue(voxelCoords + ivec3(1,1,0));
            float fPPP = getVertexValue(voxelCoords + ivec3(1,1,1));

            float a00P_000_0P0 = avg3(f00P, f000, f0P0);
            float a00P_0PP_0P0 = avg3(f00P, f0PP, f0P0);
            float a00P_000_P00 = avg3(f00P, f000, fP00);
            float a00P_P0P_P00 = avg3(f00P, fP0P, fP00);
            float aP00_0P0_00P = avg3(fP00, f0P0, f00P);
            float a0PP_P0P_PP0 = avg3(f0PP, fP0P, fPP0);
            float a0PP_0P0_PP0 = avg3(f0PP, f0P0, fPP0);
            float a0PP_PPP_PP0 = avg3(f0PP, fPPP, fPP0);
            float aP0P_P00_PP0 = avg3(fP0P, fP00, fPP0);
            float aP0P_PPP_PP0 = avg3(fP0P, fPPP, fPP0);

            float t000_00P_0P0_0PP = max3(a00P_000_0P0, a00P_0PP_0P0, f0PP);
            float t000_00P_P00_P0P = max3(a00P_000_P00, a00P_P0P_P00, f0PP);
            float t000_000_PPP_PPP = max3(aP00_0P0_00P, a0PP_P0P_PP0, fPPP);
            float t0P0_0PP_PP0_PPP = max3(a0PP_0P0_PP0, a0PP_PPP_PP0, fPPP);
            float tP00_P0P_PP0_PPP = max3(aP0P_P00_PP0, aP0P_PPP_PP0, fPPP);
            
            bool occ0 = 
            f000 >= t000_00P_0P0_0PP && 
            f000 >= t000_00P_P00_P0P && 
            f000 >= t000_000_PPP_PPP && 
            f0P0 >= t0P0_0PP_PP0_PPP && 
            fP00 >= tP00_P0P_PP0_PPP && 
            fPP0 >= fPPP;

            bool occ1 =
            mNNN >= max(f000, t000_000_PPP_PPP) &&
            m0NN >= max(f000, t000_00P_0P0_0PP) &&
            mN0N >= max(f000, t000_00P_P00_P0P) &&
            mNPN >= max(f0P0, t0P0_0PP_PP0_PPP) && 
            mPNN >= max(fP00, tP00_P0P_PP0_PPP) &&
            m00N >= max(f000, f00P) &&
            m0PN >= max(f0P0, f0PP) &&
            mP0N >= max(fP00, fP0P) &&
            mPPN >= max(fPP0, fPPP) &&
            mP0N >= fPP0 &&
            m0PN >= fPP0 &&
            m00N >= fPP0 &&
            mNN0 >= f00P &&
            mN00 >= f00P &&
            m0N0 >= f00P &&
            m00N >= f0P0 &&
            mN0N >= f0P0 &&
            m00N >= fP00 &&
            m0NN >= fP00 &&
            mNP0 >= f0PP && 
            mN00 >= f0PP && 
            mPN0 >= fP0P && 
            m0N0 >= fP0P;

            setOutput(float(occ0 || occ1));
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
    
        float avg3(float a, float b, float c) { return (a + b + c) / 3.0; }
        float max3(float a, float b, float c) { return max(max(a, b), c); }
        
        ivec3 getCellCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getMinVertexValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getB(safeCoords.z, safeCoords.y, safeCoords.x);
        }    

        float getVertexValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            ivec3 voxelCoords = cellCoords - 1;

            // Applied transformation (x,y,z) -> (1-x, 1-y, 1-z)
            float mNNN = getMinVertexValue(voxelCoords + ivec3(2,2,2));
            float mNN0 = getMinVertexValue(voxelCoords + ivec3(2,2,1));
            float mN0N = getMinVertexValue(voxelCoords + ivec3(2,1,2));
            float mN00 = getMinVertexValue(voxelCoords + ivec3(2,1,1));
            float mNPN = getMinVertexValue(voxelCoords + ivec3(2,0,2));
            float mNP0 = getMinVertexValue(voxelCoords + ivec3(2,0,1));
            float m0NN = getMinVertexValue(voxelCoords + ivec3(1,2,2));
            float m0N0 = getMinVertexValue(voxelCoords + ivec3(1,2,1));
            float m00N = getMinVertexValue(voxelCoords + ivec3(1,1,2));
            float m0PN = getMinVertexValue(voxelCoords + ivec3(1,0,2));
            float mPNN = getMinVertexValue(voxelCoords + ivec3(0,2,2));
            float mPN0 = getMinVertexValue(voxelCoords + ivec3(0,2,1));
            float mP0N = getMinVertexValue(voxelCoords + ivec3(0,1,2));
            float mPPN = getMinVertexValue(voxelCoords + ivec3(0,0,2));
            
            // Applied transformation (x,y,z) -> (1-x, 1-y, 1-z)
            float f000 = getVertexValue(voxelCoords + ivec3(1,1,1));
            float fP00 = getVertexValue(voxelCoords + ivec3(0,1,1));
            float f0P0 = getVertexValue(voxelCoords + ivec3(1,0,1));
            float f00P = getVertexValue(voxelCoords + ivec3(1,1,0));
            float f0PP = getVertexValue(voxelCoords + ivec3(1,0,0));
            float fP0P = getVertexValue(voxelCoords + ivec3(0,1,0));
            float fPP0 = getVertexValue(voxelCoords + ivec3(0,0,1));
            float fPPP = getVertexValue(voxelCoords + ivec3(0,0,0));

            float a00P_000_0P0 = avg3(f00P, f000, f0P0);
            float a00P_0PP_0P0 = avg3(f00P, f0PP, f0P0);
            float a00P_000_P00 = avg3(f00P, f000, fP00);
            float a00P_P0P_P00 = avg3(f00P, fP0P, fP00);
            float aP00_0P0_00P = avg3(fP00, f0P0, f00P);
            float a0PP_P0P_PP0 = avg3(f0PP, fP0P, fPP0);
            float a0PP_0P0_PP0 = avg3(f0PP, f0P0, fPP0);
            float a0PP_PPP_PP0 = avg3(f0PP, fPPP, fPP0);
            float aP0P_P00_PP0 = avg3(fP0P, fP00, fPP0);
            float aP0P_PPP_PP0 = avg3(fP0P, fPPP, fPP0);

            float t000_00P_0P0_0PP = max3(a00P_000_0P0, a00P_0PP_0P0, f0PP);
            float t000_00P_P00_P0P = max3(a00P_000_P00, a00P_P0P_P00, f0PP);
            float t0P0_0PP_PP0_PPP = max3(a0PP_0P0_PP0, a0PP_PPP_PP0, fPPP);
            float tP00_P0P_PP0_PPP = max3(aP0P_P00_PP0, aP0P_PPP_PP0, fPPP);
            float t000_000_PPP_PPP = max3(aP00_0P0_00P, a0PP_P0P_PP0, fPPP);

            bool occ0 = 
            f000 >= t000_00P_0P0_0PP && 
            f000 >= t000_00P_P00_P0P && 
            f000 >= t000_000_PPP_PPP && 
            f0P0 >= t0P0_0PP_PP0_PPP && 
            fP00 >= tP00_P0P_PP0_PPP && 
            fPP0 >= fPPP;

            bool occ1 =
            mNNN >= max(f000, t000_000_PPP_PPP) &&
            m0NN >= max(f000, t000_00P_0P0_0PP) &&
            mN0N >= max(f000, t000_00P_P00_P0P) &&
            mNPN >= max(f0P0, t0P0_0PP_PP0_PPP) && 
            mPNN >= max(fP00, tP00_P0P_PP0_PPP) &&
            m00N >= max(f000, f00P) &&
            m0PN >= max(f0P0, f0PP) &&
            mP0N >= max(fP00, fP0P) &&
            mPPN >= max(fPP0, fPPP) &&
            m00N >= f0P0 &&
            mN0N >= f0P0 &&
            m00N >= fP00 &&
            m0NN >= fP00 &&
            mP0N >= fPP0 &&
            m0PN >= fPP0 &&
            m00N >= fPP0 &&
            mNN0 >= f00P &&
            mN00 >= f00P &&
            m0N0 >= f00P &&
            mNP0 >= f0PP && 
            mN00 >= f0PP && 
            mPN0 >= fP0P && 
            m0N0 >= fP0P;

            setOutput(float(occ0 || occ1));
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
    const occlusionProgram   = new GPGPUBackOcclusionMap(volumeMap.shape)
    const startProgram       = new GPGPUStartPropagationMap(volumeMap.shape)

    let maxPropagation = volumeMap.shape[0]
    let propagationMap = runProgram(startProgram, [volumeMap])
    let startMap = propagationMap.clone()

    console.log(startMap.mean().dataSync())

    for (let i = 0; i < maxPropagation; i++) 
    {
        const prev = propagationMap
        propagationMap = runProgram(propagationProgram, [startMap, prev])
        prev.dispose()

        if (i%2==0) await tf.nextFrame()                     
    }

    console.log(propagationMap.mean().dataSync())

    // const occlusionMap = runProgram(occlusionProgram, [volumeMap, propagationMap], 'bool')
    // propagationMap.dispose()
    startMap.dispose()

    return propagationMap as tf.Tensor
}

async function computeFrontOcclusionMap(volumeMap: tf.Tensor3D) : Promise<tf.Tensor<tf.Rank>>
{
    const propagationProgram = new GPGPUFrontPropagationMap(volumeMap.shape)
    const occlusionProgram   = new GPGPUFrontOcclusionMap(volumeMap.shape)
    const startProgram       = new GPGPUStartPropagationMap(volumeMap.shape)

    let maxPropagation = volumeMap.shape[0]
    let propagationMap = runProgram(startProgram, [volumeMap])
    let startMap = propagationMap.clone()

    console.log(startMap.mean().dataSync())

    for (let i = 0; i < maxPropagation; i++) 
    {
        const prev = propagationMap
        propagationMap = runProgram(propagationProgram, [startMap, prev])
        prev.dispose()

        if (i%2==0) await tf.nextFrame()                     
    }

    console.log(propagationMap.mean().dataSync())

    // const occlusionMap = runProgram(occlusionProgram, [volumeMap, propagationMap], 'bool')
    // propagationMap.dispose()
    startMap.dispose()

    return propagationMap as tf.Tensor
}

export async function computeOcclusionMap(volumeMap: tf.Tensor3D) : Promise<tf.Tensor<tf.Rank>>
{
    const backOcclusionMap = await computeBackOcclusionMap(volumeMap)
    const frontOcclusionMap = await computeFrontOcclusionMap(volumeMap)

    const occlusionMap = tf.logicalOr(backOcclusionMap, frontOcclusionMap);
    tf.dispose([backOcclusionMap, frontOcclusionMap]);

    return occlusionMap as tf.Tensor
}

