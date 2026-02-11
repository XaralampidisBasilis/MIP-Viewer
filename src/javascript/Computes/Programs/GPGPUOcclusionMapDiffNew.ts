import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { unstackPacked } from './unstack_packed_keepDims_webgl'
import { stackPacked } from './stack_packed_keepDims_webgl'

type Axis = 0 | 1 | 2
type Permute = [Axis, Axis, Axis]
type Reverse = Axis[]

class GPGPUUnidirectionalDifferenceMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(inputShape: [number, number, number], permutation: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [inDepth, inHeight, inWidth] = inputShape
        this.outputShape = [inDepth, inHeight, inWidth, 2, 2]
        
        const transformVoxelOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permutation)

            for (const a of reverse) old[a] = - old[a]
            
            return old.toReversed().join(',')
        }

        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

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

        void main()
        {
            ivec3 coords = getOutCoords();

            float v111 = getA(getVoxelCoords(coords, ${transformVoxelOffset(-0,-0,-0)}));
            float v110 = getA(getVoxelCoords(coords, ${transformVoxelOffset(-0,-0,-1)}));
            float v100 = getA(getVoxelCoords(coords, ${transformVoxelOffset(-0,-1,-1)}));
            float v010 = getA(getVoxelCoords(coords, ${transformVoxelOffset(-1,-0,-1)}));
            float v000 = getA(getVoxelCoords(coords, ${transformVoxelOffset(-1,-1,-1)}));

            float d000 = v000 - v111;
            float d010 = v010 - v111;
            float d100 = v100 - v111;
            float d110 = v110 - v111;

            setOutput(vec4(d000, d010, d100, d110));
        }
        `
    }
}

class GPGPUUpdateUnidirectionalDifferenceSlices implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(outputShape: [number, number, number, 2, 2], permutation: Permute = [0,1,2], reverse: Reverse = []) 
    {
        const [outDepth, outHeight, outWidth] = outputShape.slice(0,3)
        this.outputShape = outputShape 

        const transformVoxelOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permutation)

            for (const a of reverse) old[a] = - old[a]
            
            return old.toReversed().join(',')
        }

        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        float min4(float a, float b, float c, float d) { return min(min(min(a, b), c), d); }
        float min4(vec4 v) { return min(min(min(v.x, v.y), v.z), v.w); }

        float avg4(float a, float b, float c, float d) { return (a + b + c + d) * 0.25; }
        float avg2(float a, float b) { return (a + b) * 0.5; }

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        ivec3 getVoxelCoords(ivec3 coords, int ox, int oy, int oz)
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

        void main()
        {
            ivec3 coords = getOutCoords();

            vec4 d111 = getA(getVoxelCoords(coords, ${transformVoxelOffset(-0,-0,-0)}));
            vec4 d110 = getB(getVoxelCoords(coords, ${transformVoxelOffset(-0,-0,-1)}));
            vec4 d100 = getB(getVoxelCoords(coords, ${transformVoxelOffset(-0,-1,-1)}));
            vec4 d010 = getB(getVoxelCoords(coords, ${transformVoxelOffset(-1,-0,-1)}));
            vec4 d000 = getB(getVoxelCoords(coords, ${transformVoxelOffset(-1,-1,-1)}));

            d111.x += max(min4(d000), 0.0);
            d111.y += max(min4(d010), 0.0);
            d111.z += max(min4(d100), 0.0);
            d111.w += max(min4(d110), 0.0);

            setOutput(d111);
        }
        `
    }
}

class GPGPUUpdateUnidirectionalDifferenceMap implements GPGPUProgram 
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

        const transformVoxelOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permutation)

            for (const a of reverse) old[a] = - old[a]
            
            return old.toReversed().join(',')
        }

        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        float min4(float a, float b, float c, float d) { return min(min(min(a, b), c), d); }
        float min4(vec4 v) { return min(min(min(v.x, v.y), v.z), v.w); }

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        ivec3 getVoxelCoords(ivec3 coords, int ox, int oy, int oz)
        {
            return coords + ivec3(ox, oy, oz);
        }

        vec4 getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x, 0, 0);
        }

        void main()
        {
            ivec3 coords = getOutCoords();

            vec4 d111 = getA(getVoxelCoords(coords, ${transformVoxelOffset(-0,-0,-0)}));
            vec4 d110 = getA(getVoxelCoords(coords, ${transformVoxelOffset(-0,-0,-1)}));
            vec4 d100 = getA(getVoxelCoords(coords, ${transformVoxelOffset(-0,-1,-1)}));
            vec4 d010 = getA(getVoxelCoords(coords, ${transformVoxelOffset(-1,-0,-1)}));
            vec4 d000 = getA(getVoxelCoords(coords, ${transformVoxelOffset(-1,-1,-1)}));

            d111.x += max(min4(d000), 0.0);
            d111.y += max(min4(d010), 0.0);
            d111.z += max(min4(d100), 0.0);
            d111.w += max(min4(d110), 0.0);

            setOutput(d111);
        }
        `
    }
}

class GPGPUUnidirectionalMinimaMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
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
        float min4(vec4 v) { return min(min(min(v.x, v.y), v.z), v.w); }

        struct CellValues 
        { 
            float d000; 
            float d100; 
            float d010; 
            float d001; 
            float d011; 
            float d101; 
            float d110; 
            float d111; 
        }; 

        ivec3 getOutCoords()
        {
            ivec5 cCoords = getOutputCoords();
            return ivec3(cCoords.z, cCoords.y, cCoords.x);
        }

        ivec3 getVoxelCoords(ivec3 vCoords, int ox, int oy, int oz)
        {
            return vCoords + ivec3(ox, oy, oz);
        }

        vec4 getA(ivec3 vCoords)
        {
            vCoords = clamp(vCoords, minCoords, maxCoords);
            return getA(vCoords.z, vCoords.y, vCoords.x, 0, 0);
        }

        float getMinPosA(ivec3 vCoords)
        {
            return max(min4(getA(vCoords)), 0.0);
        }

        CellValues getValues(ivec3 cCoords)
        {
            CellValues c;

            ivec3 vCoords = cCoords - 1;

            c.d000 = getMinPosA(getVoxelCoords(vCoords, ${transformVoxelOffset(0,0,0)}));
            c.d100 = getMinPosA(getVoxelCoords(vCoords, ${transformVoxelOffset(1,0,0)}));
            c.d010 = getMinPosA(getVoxelCoords(vCoords, ${transformVoxelOffset(0,1,0)}));
            c.d001 = getMinPosA(getVoxelCoords(vCoords, ${transformVoxelOffset(0,0,1)}));
            c.d011 = getMinPosA(getVoxelCoords(vCoords, ${transformVoxelOffset(0,1,1)}));
            c.d101 = getMinPosA(getVoxelCoords(vCoords, ${transformVoxelOffset(1,0,1)}));
            c.d110 = getMinPosA(getVoxelCoords(vCoords, ${transformVoxelOffset(1,1,0)}));
            c.d111 = getMinPosA(getVoxelCoords(vCoords, ${transformVoxelOffset(1,1,1)}));

            return c;
        }

        float getMinDifferenceOnFaceX(CellValues c)
        {           
            return min4(c.d000, c.d010, c.d001, c.d011);
        }

        float getMinDifferenceOnFaceY(CellValues c)
        {
            return min4(c.d000, c.d100, c.d001, c.d101);
        }
            
        float getMinDifferenceOnFaceZ(CellValues c)
        {
            return min4(c.d000, c.d010, c.d100, c.d110);
        }

        void main()
        {
            ivec3 cCoords = getOutCoords();
            CellValues c = getValues(cCoords);

            float xMin = getMinDifferenceOnFaceX(c);
            float yMin = getMinDifferenceOnFaceY(c);
            float zMin = getMinDifferenceOnFaceZ(c);

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

            c.v000 = getA(getVoxelCoords(coords, ${transformVoxelOffset(0,0,0)}));
            c.v100 = getA(getVoxelCoords(coords, ${transformVoxelOffset(1,0,0)}));
            c.v010 = getA(getVoxelCoords(coords, ${transformVoxelOffset(0,1,0)}));
            c.v001 = getA(getVoxelCoords(coords, ${transformVoxelOffset(0,0,1)}));
            c.v011 = getA(getVoxelCoords(coords, ${transformVoxelOffset(0,1,1)}));
            c.v101 = getA(getVoxelCoords(coords, ${transformVoxelOffset(1,0,1)}));
            c.v110 = getA(getVoxelCoords(coords, ${transformVoxelOffset(1,1,0)}));
            c.v111 = getA(getVoxelCoords(coords, ${transformVoxelOffset(1,1,1)}));

            return c;
        }

        float getMaxDifferenceOnFaceX(CellValues c)
        {
            float m = -1.0;

            m = max(m, avg3(c.v001, c.v010, c.v011) - c.v000);
            m = max(m, avg3(c.v001, c.v010, c.v100) - c.v000);
            m = max(m, avg3(c.v001, c.v100, c.v101) - c.v000);
            m = max(m, avg3(c.v011, c.v101, c.v110) - c.v000);
            m = max(m, avg3(c.v001, c.v010, c.v000) - c.v000);
            m = max(m, avg3(c.v001, c.v100, c.v000) - c.v000);
            m = max(m, avg3(c.v011, c.v110, c.v111) - c.v010);
            m = max(m, avg3(c.v011, c.v110, c.v010) - c.v010);
            m = max(m, c.v001 - c.v000);
            m = max(m, c.v011 - c.v000);
            m = max(m, c.v101 - c.v000);
            m = max(m, c.v111 - c.v000);
            m = max(m, c.v011 - c.v010);
            m = max(m, c.v111 - c.v010);

            return m;
        }
    
        float getMaxDifferenceOnFaceY(CellValues c)
        {
            float m = -1.0;

            m = max(m, avg3(c.v001, c.v010, c.v011) - c.v000);
            m = max(m, avg3(c.v001, c.v010, c.v100) - c.v000);
            m = max(m, avg3(c.v001, c.v100, c.v101) - c.v000);
            m = max(m, avg3(c.v011, c.v101, c.v110) - c.v000);
            m = max(m, avg3(c.v001, c.v010, c.v000) - c.v000);
            m = max(m, avg3(c.v001, c.v100, c.v000) - c.v000);
            m = max(m, avg3(c.v101, c.v110, c.v111) - c.v100);
            m = max(m, avg3(c.v101, c.v110, c.v100) - c.v100);
            m = max(m, c.v001 - c.v000);
            m = max(m, c.v011 - c.v000);
            m = max(m, c.v101 - c.v000);
            m = max(m, c.v111 - c.v000);
            m = max(m, c.v101 - c.v100);
            m = max(m, c.v111 - c.v100);

            return m;
        }

        float getMaxDifferenceOnFaceZ(CellValues c)
        {
            float m = -1.0;

            m = max(m, avg3(c.v001, c.v010, c.v011) - c.v000);
            m = max(m, avg3(c.v001, c.v010, c.v100) - c.v000);
            m = max(m, avg3(c.v001, c.v100, c.v101) - c.v000);
            m = max(m, avg3(c.v011, c.v101, c.v110) - c.v000);
            m = max(m, avg3(c.v001, c.v010, c.v000) - c.v000);
            m = max(m, avg3(c.v001, c.v100, c.v000) - c.v000);
            m = max(m, avg3(c.v011, c.v110, c.v111) - c.v010);
            m = max(m, avg3(c.v011, c.v110, c.v010) - c.v010);
            m = max(m, avg3(c.v101, c.v110, c.v111) - c.v100);
            m = max(m, avg3(c.v101, c.v110, c.v100) - c.v100);
            m = max(m, c.v001 - c.v000);
            m = max(m, c.v011 - c.v000);
            m = max(m, c.v101 - c.v000);
            m = max(m, c.v111 - c.v000);
            m = max(m, c.v011 - c.v010);
            m = max(m, c.v111 - c.v010);
            m = max(m, c.v101 - c.v100);
            m = max(m, c.v111 - c.v100);
            m = max(m, c.v111 - c.v110);

            return m;
        }

        void main()
        {
            ivec3 coords = getOutCoords();
            CellValues c = getValues(coords);

            float xMax = getMaxDifferenceOnFaceX(c);
            float yMax = getMaxDifferenceOnFaceY(c);
            float zMax = getMaxDifferenceOnFaceZ(c);

            setOutput(vec4(xMax, yMax, zMax, 0.0));
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
            bool occlusion = all(tests.xyz);

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

// ground truth

export function computeUnidirectionalOcclusionMapBase(volume: tf.Tensor3D, permutation: Permute, reverse: Reverse, verbose: boolean = false) : tf.Tensor3D
{
    const reversed = volume.reverse(reverse) as tf.Tensor3D
    const transposed = reversed.transpose(permutation) as tf.Tensor3D
    reversed.dispose()

    const differenceProgram = new GPGPUUnidirectionalDifferenceMap(volume.shape)
    const differenceStack = runWebGLProgram(differenceProgram, [transposed], 'float32', [], true)
    if (verbose) logTensor('differenceStack', differenceStack)

    const slices = unstackPacked(differenceStack, 0) 
    differenceStack.dispose()

    const sliceShape = slices[0].shape as [number, number, number, 2, 2]
    const sliceProgram = new GPGPUUpdateUnidirectionalDifferenceSlices(sliceShape)

    for (let i = 1; i < slices.length; i++)
    {
        const t = runWebGLProgram(sliceProgram, [slices[i], slices[i-1]], 'float32', [[i]], true)
        tf.dispose(slices[i])
        slices[i] = t
    }
        
    const difference = stackPacked(slices, 0) 
    tf.dispose(slices)
    if (verbose) logTensor('difference', difference)

    const minimaProgram = new GPGPUUnidirectionalMinimaMap(volume.shape)
    const minima = runWebGLProgram(minimaProgram, [difference], 'float32', [], true)
    tf.dispose(difference)
    if (verbose) logTensor('minima', minima)

    const maximaProgram = new GPGPUUnidirectionalMaximaMap(volume.shape)
    const maxima = runWebGLProgram(maximaProgram, [transposed], 'float32', [], true)
    tf.dispose(transposed)
    if (verbose) logTensor('maxima', maxima)

    const occlusionMapShape = minima.shape.slice(0,3) as [number, number, number]
    const occlusionProgram = new GPGPUUnidirectionalOcclusionMap(occlusionMapShape)
    const occlusion = runWebGLProgram(occlusionProgram, [minima, maxima], 'float32', [], true) as tf.Tensor3D
    tf.dispose([minima, maxima])
    if (verbose) logTensor('occlusion', occlusion)

    const untransposed = occlusion.transpose(inversePermutation(permutation))
    tf.dispose(occlusion)

    const unreversed = untransposed.reverse(reverse)
    tf.dispose(untransposed)

    return unreversed as tf.Tensor3D
}

export function computeBidirectionalOcclusionMapBase(volumeMap: tf.Tensor3D, permutation: Permute, reverse: Reverse, verbose: boolean = false) : tf.Tensor3D
{
    const occlusionMaps = [
        computeUnidirectionalOcclusionMapBase(volumeMap, permutation, reverse),
        computeUnidirectionalOcclusionMapBase(volumeMap, permutation, complementReverse(reverse)),
    ]

    const bidirectionalProgram = new GPGPUBidirectionalOcclusionMap(occlusionMaps[0].shape)
    const occlusionMap = runWebGLProgram(bidirectionalProgram, occlusionMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)
    if (verbose) logTensor('bidirectionalOcclusion', occlusionMap)

    return occlusionMap as tf.Tensor3D
}

export function computeAnisotropicBidirectionalOcclusionMapBase(volumeMap: tf.Tensor3D, permutation: Permute, verbose: boolean = false) : tf.Tensor3D
{
    const reverseA = permutation.slice(1, 1) as Reverse
    const reverseB = permutation.slice(1, 2) as Reverse
    const reverseC = permutation.slice(2, 3) as Reverse
    const reverseD = permutation.slice(1, 3) as Reverse

    const occlusionMaps = [
        computeBidirectionalOcclusionMapBase(volumeMap, permutation, reverseA),
        computeBidirectionalOcclusionMapBase(volumeMap, permutation, reverseB),
        computeBidirectionalOcclusionMapBase(volumeMap, permutation, reverseC),
        computeBidirectionalOcclusionMapBase(volumeMap, permutation, reverseD),
    ]

    const anisotropicBidirectionalProgram = new GPGPUAnisotropicBidirectionalOcclusionMap(occlusionMaps[0].shape)
    const occlusionMap = runWebGLProgram(anisotropicBidirectionalProgram, occlusionMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)

    if (verbose) logAnisotropicBidirectionalOcclusionMaps(occlusionMap)

    return occlusionMap 
}

export function computeExtendedAnisotropicBidirectionalOcclusionMapBase(volumeMap: tf.Tensor3D, verbose: boolean = false) : tf.Tensor3D
{
    const permuteX = [2,1,0] as Permute
    const permuteY = [1,2,0] as Permute
    const permuteZ = [0,1,2] as Permute

    const occlusionMaps = [
        computeAnisotropicBidirectionalOcclusionMapBase(volumeMap, permuteX),
        computeAnisotropicBidirectionalOcclusionMapBase(volumeMap, permuteY),
        computeAnisotropicBidirectionalOcclusionMapBase(volumeMap, permuteZ),
    ]

    const extendedAnisotropicBidirectionalProgram = new GPGPUExtendedAnisotropicBidirectionalOcclusionMap(occlusionMaps[0].shape)
    const occlusionMap = runWebGLProgram(extendedAnisotropicBidirectionalProgram, occlusionMaps, 'int32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)

    if (verbose) logExtendedAnisotropicBidirectionalOcclusionMaps(occlusionMap)

    return occlusionMap 
}

// sync functions 

export function computeUnidirectionalOcclusionMap(volume: tf.Tensor3D, permutation: Permute, reverse: Reverse, verbose: boolean = false) : tf.Tensor3D
{
    const axis = permutation[0]
    const reverseAxis = reverse.includes(axis)

    const differenceProgram = new GPGPUUnidirectionalDifferenceMap(volume.shape, permutation, reverse)
    const differenceStack = runWebGLProgram(differenceProgram, [volume], 'float32', [], true)
    if (verbose) logTensor('differenceStack', differenceStack)

    const slices = unstackPacked(differenceStack, axis) 
    differenceStack.dispose()
    if (reverseAxis) slices.reverse()

    const sliceShape = slices[0].shape as [number, number, number, 2, 2]
    const sliceProgram = new GPGPUUpdateUnidirectionalDifferenceSlices(sliceShape, permutation, reverse)

    for (let i = 1; i < slices.length; i++)
    {
        const t = runWebGLProgram(sliceProgram, [slices[i], slices[i-1]], 'float32', [[i]], true)
        tf.dispose(slices[i])
        slices[i] = t
    }

    if (reverseAxis) slices.reverse()
    const difference = stackPacked(slices, axis) 
    tf.dispose(slices)
    if (verbose) logTensor('difference', difference)

    const minimaProgram = new GPGPUUnidirectionalMinimaMap(volume.shape, permutation, reverse)
    const minima = runWebGLProgram(minimaProgram, [difference], 'float32', [], true)
    tf.dispose(difference)
    if (verbose) logTensor('minima', minima)

    const maximaProgram = new GPGPUUnidirectionalMaximaMap(volume.shape, permutation, reverse)
    const maxima = runWebGLProgram(maximaProgram, [volume], 'float32', [], true)
    if (verbose) logTensor('maxima', maxima)

    const occlusionShape = minima.shape.slice(0,3) as [number, number, number]
    const occlusionProgram = new GPGPUUnidirectionalOcclusionMap(occlusionShape)
    const occlusion = runWebGLProgram(occlusionProgram, [minima, maxima], 'float32', [], true) as tf.Tensor3D
    tf.dispose([minima, maxima])
    if (verbose) logTensor('occlusion', occlusion)

    return occlusion as tf.Tensor3D
}

export function computeBidirectionalOcclusionMap(volumeMap: tf.Tensor3D, permutation: Permute, reverse: Reverse, verbose: boolean = false) : tf.Tensor3D
{
    const occlusionMaps = [
        computeUnidirectionalOcclusionMap(volumeMap, permutation, reverse),
        computeUnidirectionalOcclusionMap(volumeMap, permutation, complementReverse(reverse)),
    ]

    const bidirectionalProgram = new GPGPUBidirectionalOcclusionMap(occlusionMaps[0].shape)
    const occlusionMap = runWebGLProgram(bidirectionalProgram, occlusionMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)

    if (verbose) logTensor('bidirectionalOcclusion', occlusionMap)

    return occlusionMap as tf.Tensor3D
}

export function computeAnisotropicBidirectionalOcclusionMap(volumeMap: tf.Tensor3D, permutation: Permute, verbose: boolean = false) : tf.Tensor3D
{
    const reverseA = permutation.slice(1, 1) as Reverse
    const reverseB = permutation.slice(1, 2) as Reverse
    const reverseC = permutation.slice(2, 3) as Reverse
    const reverseD = permutation.slice(1, 3) as Reverse

    const occlusionMaps = [
        computeBidirectionalOcclusionMap(volumeMap, permutation, reverseA),
        computeBidirectionalOcclusionMap(volumeMap, permutation, reverseB),
        computeBidirectionalOcclusionMap(volumeMap, permutation, reverseC),
        computeBidirectionalOcclusionMap(volumeMap, permutation, reverseD),
    ]

    const anisotropicBidirectionalProgram = new GPGPUAnisotropicBidirectionalOcclusionMap(occlusionMaps[0].shape)
    const occlusionMap = runWebGLProgram(anisotropicBidirectionalProgram, occlusionMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)

    if (verbose) logAnisotropicBidirectionalOcclusionMaps(occlusionMap)

    return occlusionMap 
}

export function computeExtendedAnisotropicBidirectionalOcclusionMap(volumeMap: tf.Tensor3D, verbose: boolean = false) : tf.Tensor3D
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
    const occlusionMap = runWebGLProgram(extendedAnisotropicBidirectionalProgram, occlusionMaps, 'int32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)

    if (verbose) logExtendedAnisotropicBidirectionalOcclusionMaps(occlusionMap)

    return occlusionMap 
}

// async functions

export async function computeUnidirectionalOcclusionMapAsync(volume: tf.Tensor3D, permutation: Permute, reverse: Reverse, verbose: boolean = false) : Promise<tf.Tensor3D>
{
    const minimaProgram = new GPGPUUnidirectionalDifferenceMap(volume.shape, permutation, reverse)
    let minima = runWebGLProgram(minimaProgram, [volume], 'float32', [], true)
    if (verbose) logTensor('minimaRaw', minima)

    const minimaShape = minima.shape as [number, number, number, 2, 2]
    const updateProgram = new GPGPUUpdateUnidirectionalDifferenceMap(minimaShape, permutation, reverse)

    for (let i = 1; i < minimaShape[permutation[0]]; i++)
    {
        const map = runWebGLProgram(updateProgram, [minima], 'float32', [], true)
        tf.dispose(minima)
        minima = map

        await tf.nextFrame()
    }
    if (verbose) logTensor('minima', minima)

    const maximaProgram = new GPGPUUnidirectionalMaximaMap(volume.shape, permutation, reverse)
    const maxima = runWebGLProgram(maximaProgram, [volume], 'float32', [], true)
    if (verbose) logTensor('maxima', maxima)

    const occlusionShape = minimaShape.slice(0,3) as [number, number, number]
    const occlusionProgram = new GPGPUUnidirectionalOcclusionMap(occlusionShape)
    const occlusion = runWebGLProgram(occlusionProgram, [minima, maxima], 'float32', [], true)
    tf.dispose([minima, maxima])
    if (verbose) logTensor('occlusion', occlusion)

    return occlusion as tf.Tensor3D
}

export async function computeBidirectionalOcclusionMapAsync(volumeMap: tf.Tensor3D, permutation: Permute, reverse: Reverse, verbose: boolean = false) : Promise<tf.Tensor3D>
{
    const occlusionMaps = [
        await computeUnidirectionalOcclusionMapAsync(volumeMap, permutation, reverse),
        await computeUnidirectionalOcclusionMapAsync(volumeMap, permutation, complementReverse(reverse)),
    ]

    const bidirectionalProgram = new GPGPUBidirectionalOcclusionMap(occlusionMaps[0].shape)
    const occlusionMap = runWebGLProgram(bidirectionalProgram, occlusionMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)
    if (verbose) logTensor('bidirectionalOcclusion', occlusionMap)

    return occlusionMap as tf.Tensor3D
}

export async function computeAnisotropicBidirectionalOcclusionMapAsync(volumeMap: tf.Tensor3D, permutation: Permute, verbose: boolean = false) : Promise<tf.Tensor3D>
{
    const reverseA = permutation.slice(1, 1) as Reverse
    const reverseB = permutation.slice(1, 2) as Reverse
    const reverseC = permutation.slice(2, 3) as Reverse
    const reverseD = permutation.slice(1, 3) as Reverse

    const occlusionMaps = [
        await computeBidirectionalOcclusionMapAsync(volumeMap, permutation, reverseA),
        await computeBidirectionalOcclusionMapAsync(volumeMap, permutation, reverseB),
        await computeBidirectionalOcclusionMapAsync(volumeMap, permutation, reverseC),
        await computeBidirectionalOcclusionMapAsync(volumeMap, permutation, reverseD),
    ]

    const anisotropicBidirectionalProgram = new GPGPUAnisotropicBidirectionalOcclusionMap(occlusionMaps[0].shape)
    const occlusionMap = runWebGLProgram(anisotropicBidirectionalProgram, occlusionMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)

    if (verbose) logAnisotropicBidirectionalOcclusionMaps(occlusionMap)

    return occlusionMap 
}

export async function computeExtendedAnisotropicBidirectionalOcclusionMapAsync(volumeMap: tf.Tensor3D, verbose: boolean = false) : Promise<tf.Tensor3D>
{
    const permuteX = [2,1,0] as Permute
    const permuteY = [1,0,2] as Permute
    const permuteZ = [0,1,2] as Permute

    const occlusionMaps = [
        await computeAnisotropicBidirectionalOcclusionMapAsync(volumeMap, permuteX),
        await computeAnisotropicBidirectionalOcclusionMapAsync(volumeMap, permuteY),
        await computeAnisotropicBidirectionalOcclusionMapAsync(volumeMap, permuteZ),
    ]

    const extendedAnisotropicBidirectionalProgram = new GPGPUExtendedAnisotropicBidirectionalOcclusionMap(occlusionMaps[0].shape)
    const occlusionMap = runWebGLProgram(extendedAnisotropicBidirectionalProgram, occlusionMaps, 'float32', [], true) as tf.Tensor3D
    tf.dispose(occlusionMaps)

    if (verbose) logExtendedAnisotropicBidirectionalOcclusionMaps(occlusionMap)

    return occlusionMap 
}

// debug

export function computeExtendedAnisotropicBidirectionalOcclusionMapDebug(volumeMap: tf.Tensor3D, verbose: boolean = false) : tf.Tensor3D
{
    let o1,o2,o3,o4,ox,oy,oz

    // let t = computeUnidirectionalOcclusionMap(volumeMap, [0,1,2], [])
    // let t = computeUnidirectionalOcclusionMap(volumeMap, [0,1,2], [0,1,2])
    let t = computeUnidirectionalOcclusionMap(volumeMap,  [2,1,0], [], true)

    o1 = tf.onesLike(t) // computeBidirectionalOcclusionMap(volumeMap, [2,1,0], [   ])
    o2 = tf.onesLike(t) // computeBidirectionalOcclusionMap(volumeMap, [2,1,0], [  1])
    o3 = tf.onesLike(t) // computeBidirectionalOcclusionMap(volumeMap, [2,1,0], [  0])
    o4 = tf.onesLike(t) // computeBidirectionalOcclusionMap(volumeMap, [2,1,0], [1,0])

    ox = runWebGLProgram(new GPGPUAnisotropicBidirectionalOcclusionMap(t.shape), [o1,o2,o3,o4], 'float32', [], true) as tf.Tensor3D
    tf.dispose([o1,o2,o3,o4])
   
    o1 = tf.onesLike(t) // computeBidirectionalOcclusionMap(volumeMap, [1,2,0], [   ])
    o2 = tf.onesLike(t) // computeBidirectionalOcclusionMap(volumeMap, [1,2,0], [  2])
    o3 = tf.onesLike(t) // computeBidirectionalOcclusionMap(volumeMap, [1,2,0], [  0])
    o4 = tf.onesLike(t) // computeBidirectionalOcclusionMap(volumeMap, [1,2,0], [2,0])

    oy = runWebGLProgram(new GPGPUAnisotropicBidirectionalOcclusionMap(t.shape), [o1,o2,o3,o4], 'float32', [], true) as tf.Tensor3D
    tf.dispose([o1,o2,o3,o4])

    o1 = tf.clone(t) // computeBidirectionalOcclusionMap(volumeMap, [0,1,2], [   ])
    o2 = tf.onesLike(t) // computeBidirectionalOcclusionMap(volumeMap, [0,1,2], [  1])
    o3 = tf.onesLike(t) // computeBidirectionalOcclusionMap(volumeMap, [0,1,2], [  2])
    o4 = tf.onesLike(t) // computeBidirectionalOcclusionMap(volumeMap, [0,1,2], [1,2])

    oz = runWebGLProgram(new GPGPUAnisotropicBidirectionalOcclusionMap(t.shape), [o1,o2,o3,o4], 'float32', [], true) as tf.Tensor3D
    tf.dispose([o1,o2,o3,o4])


    const o = runWebGLProgram(new GPGPUExtendedAnisotropicBidirectionalOcclusionMap(t.shape), [ox,oy,oz], 'int32', [], true) as tf.Tensor3D
    tf.dispose([ox,oy,oz])

    if (verbose) logExtendedAnisotropicBidirectionalOcclusionMaps(o)

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
    dtype?: tf.DataType, 
    uniforms?: number[][], 
    preventEagerUnpackingOfOutput?: boolean): tf.Tensor
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(prog, inputs, dtype, uniforms, preventEagerUnpackingOfOutput)
    return tf.engine().makeTensorFromTensorInfo(info) 
}
