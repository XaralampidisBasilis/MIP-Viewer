import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { unstackPacked } from './unstack_packed_keepDims_webgl'
import { stackPacked } from './stack_packed_keepDims_webgl'
import {
    type Axis,
    type Octant,
    type Permute,
    type Reverse,
    applyPermutation,
    complementReverse,
    inversePermutation,
    permuteReverseFromDominantAxisOctant,
    reverseOctant,
} from './ShadowMapUtils'

type Array3<T> = [T, T, T]
type Array4<T> = [T, T, T, T]

class UnidirectionalMinimaMapHollow implements GPGPUProgram 
{
    variableNames = ['A', 'B']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        volumeShape: [number, number, number], 
        permute: Permute = [0,1,2], 
        reverse: Reverse = []
    ) {
        
        const transformVoxelOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)
            for (const a of reverse) old[a] = 1 - old[a]
            return old.toReversed().join(',')
        }

        const [inDepth, inHeight, inWidth] = volumeShape
        const [outDepth, outHeight, outWidth] = volumeShape.map((x) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]  
        this.userCode = `
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

        float min4(float a, float b, float c, float d) 
        {
            return min(min(min(a, b), c), d); 
        }

        bool inBounds(ivec3 coords)
        {
            return  all(greaterThanEqual(coords, minCoords)) && 
                    all(lessThanEqual(coords, maxCoords));
        }

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();

            return ivec3(coords.z, coords.y, coords.x);
        }

        float getAAt(ivec3 coords)
        {
            if (inBounds(coords)) 
                return getA(coords.z, coords.y, coords.x);
            else 
                return 0.0;
        }

        float getBAt(ivec3 coords)
        {
            if (inBounds(coords)) 
                return getB(coords.z, coords.y, coords.x);
            else 
                return 0.0;
        }

        bool isHollow(ivec3 coords)
        {
            return (getBAt(coords) > 0.5);
        }

        CellValues getValues(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            CellValues c;
            c.v000 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(0,0,0)}));
            c.v100 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(1,0,0)}));
            c.v010 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(0,1,0)}));
            c.v001 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(0,0,1)}));
            c.v011 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(0,1,1)}));
            c.v101 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(1,0,1)}));
            c.v110 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(1,1,0)}));
            c.v111 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(1,1,1)}));

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
            ivec3 coords = getOutCoords();
            CellValues cellValues = getValues(coords);

            float xMin = getMinOnFaceX(cellValues);
            float yMin = getMinOnFaceY(cellValues);
            float zMin = getMinOnFaceZ(cellValues);

            if (isHollow(coords))
                setOutput(vec4(0.0, 0.0, 0.0, 1.0));
            else
                setOutput(vec4(xMin, yMin, zMin, 0.0));
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

    constructor(
        volumeShape: [number, number, number], 
        permute: Permute = [0,1,2], 
        reverse: Reverse = []
    ) {
        
        const transformVoxelOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)
            for (const a of reverse) old[a] = 1 - old[a]
            return old.toReversed().join(',')
        }

        const [inDepth, inHeight, inWidth] = volumeShape
        const [outDepth, outHeight, outWidth] = volumeShape.map((x) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]  
        this.userCode = `
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

        float min4(float a, float b, float c, float d) 
        {
            return min(min(min(a, b), c), d); 
        }

        bool inBounds(ivec3 coords)
        {
            return  all(greaterThanEqual(coords, minCoords)) && 
                    all(lessThanEqual(coords, maxCoords));
        }

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();

            return ivec3(coords.z, coords.y, coords.x);
        }

        float getAAt(ivec3 coords)
        {
            if (inBounds(coords)) 
                return getA(coords.z, coords.y, coords.x);
            else 
                return 0.0;
        }

        CellValues getValues(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            CellValues c;
            c.v000 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(0,0,0)}));
            c.v100 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(1,0,0)}));
            c.v010 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(0,1,0)}));
            c.v001 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(0,0,1)}));
            c.v011 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(0,1,1)}));
            c.v101 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(1,0,1)}));
            c.v110 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(1,1,0)}));
            c.v111 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(1,1,1)}));

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
            ivec3 coords = getOutCoords();
            CellValues cellValues = getValues(coords);

            float xMin = getMinOnFaceX(cellValues);
            float yMin = getMinOnFaceY(cellValues);
            float zMin = getMinOnFaceZ(cellValues);

            setOutput(vec4(xMin, yMin, zMin, 0.0));
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

    constructor(
        volumeShape: [number, number, number], 
        permute: Permute = [0,1,2], 
        reverse: Reverse = []
    ) {
        
        const transformVoxelOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)
            for (const a of reverse) old[a] = 1 - old[a]
            return old.toReversed().join(',')
        }

        const [inDepth, inHeight, inWidth] = volumeShape
        const [outDepth, outHeight, outWidth] = volumeShape.map((x) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]     
        this.userCode = `
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

        float max4(float a, float b, float c, float d) 
        { 
            return max(max(max(a, b), c), d); 
        }

        bool inBounds(ivec3 coords)
        {
            return  all(greaterThanEqual(coords, minCoords)) && 
                    all(lessThanEqual(coords, maxCoords));
        }

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            
            return ivec3(coords.z, coords.y, coords.x);
        }

        float getAAt(ivec3 coords)
        {
            if (inBounds(coords)) 
                return getA(coords.z, coords.y, coords.x);
            else
                return 0.0;
        }

        CellValues getValues(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;
            
            CellValues c;
            c.v000 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(0,0,0)}));
            c.v100 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(1,0,0)}));
            c.v010 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(0,1,0)}));
            c.v001 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(0,0,1)}));
            c.v011 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(0,1,1)}));
            c.v101 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(1,0,1)}));
            c.v110 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(1,1,0)}));
            c.v111 = getAAt(voxelCoords + ivec3(${transformVoxelOffset(1,1,1)}));

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
            ivec3 coords = getOutCoords();
            CellValues cellValues = getValues(coords);

            float xMax = getMaxOnFaceX(cellValues);
            float yMax = getMaxOnFaceY(cellValues);
            float zMax = getMaxOnFaceZ(cellValues);

            setOutput(vec4(xMax, yMax, zMax, 0.5));
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

    customUniforms = [{ name: 'slice', type: 'int' as const }]

    constructor(
        outputShape: [number, number, number, 2, 2], 
        permute: Permute = [0,1,2], 
        reverse: Reverse = []
    ) {
        
        const transformCellOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)
            for (const a of reverse) old[a] = - old[a]

            const axis = permute[0]
            old[axis] = 0
            
            return old.toReversed().join(',')
        }

        const [outDepth, outHeight, outWidth] = outputShape.slice(0, 3)
        this.outputShape = outputShape 
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        bool inBounds(ivec3 coords)
        {
            return  all(greaterThanEqual(coords, minCoords)) && 
                    all(lessThanEqual(coords, maxCoords));
        }

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();

            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getAAt(ivec3 coords)
        {
            if (inBounds(coords)) 
                return getA(coords.z, coords.y, coords.x, 0, 0);
            else
                return vec4(0.0);
        }

        vec4 getBAt(ivec3 coords)
        {
            if (inBounds(coords)) 
                return getB(coords.z, coords.y, coords.x, 0, 0);
            else
                return vec4(0.0);    
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

            vec4 c111 = getAAt(coords + ivec3(${transformCellOffset(-0,-0,-0)}));
            vec4 c011 = getAAt(coords + ivec3(${transformCellOffset(-1,-0,-0)}));
            vec4 c101 = getAAt(coords + ivec3(${transformCellOffset(-0,-1,-0)}));
            vec4 c001 = getAAt(coords + ivec3(${transformCellOffset(-1,-1,-0)}));
            vec4 c110 = getBAt(coords + ivec3(${transformCellOffset(-0,-0,-1)}));
            vec4 c010 = getBAt(coords + ivec3(${transformCellOffset(-1,-0,-1)}));
            vec4 c100 = getBAt(coords + ivec3(${transformCellOffset(-0,-1,-1)}));
            vec4 c000 = getBAt(coords + ivec3(${transformCellOffset(-1,-1,-1)}));

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

    constructor(
        outputShape: [number, number, number, 2, 2], 
        permute: Permute = [0,1,2], 
        reverse: Reverse = []
    ) {

        const transformCellOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)
            for (const a of reverse) old[a] = - old[a]
            return old.toReversed().join(',')
        }

        const [outDepth, outHeight, outWidth] = outputShape.slice(0,3)
        this.outputShape = outputShape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        bool inBounds(ivec3 coords)
        {
            return  all(greaterThanEqual(coords, minCoords)) && 
                    all(lessThanEqual(coords, maxCoords));
        }

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();

            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getAAt(ivec3 coords)
        {
            if (inBounds(coords)) 
                return getA(coords.z, coords.y, coords.x, 0, 0);
            else
                return vec4(0.0);
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

            vec4 c111 = getAAt(coords + ivec3(${transformCellOffset(-0,-0,-0)}));
            vec4 c011 = getAAt(coords + ivec3(${transformCellOffset(-1,-0,-0)}));
            vec4 c101 = getAAt(coords + ivec3(${transformCellOffset(-0,-1,-0)}));
            vec4 c001 = getAAt(coords + ivec3(${transformCellOffset(-1,-1,-0)}));
            vec4 c110 = getAAt(coords + ivec3(${transformCellOffset(-0,-0,-1)}));
            vec4 c010 = getAAt(coords + ivec3(${transformCellOffset(-1,-0,-1)}));
            vec4 c100 = getAAt(coords + ivec3(${transformCellOffset(-0,-1,-1)}));
            vec4 c000 = getAAt(coords + ivec3(${transformCellOffset(-1,-1,-1)}));

            c111.x = getMinOnFaceX(c111, c110, c101, c100);
            c111.y = getMinOnFaceY(c111, c110, c011, c010);
            c111.z = getMinOnFaceZ(c111, c110, c101, c011, c100, c010, c001, c000);

            setOutput(c111);
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

    customUniforms = [{ name: 'tolerance', type: 'float' as const }]

    constructor(
        outputShape: [number, number, number], 
        permute: Permute = [0,1,2], 
        reverse: Reverse = []
    ) {

        const transformCellOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)
            for (const a of reverse) old[a] = - old[a]
            return old.toReversed().join(',')
        }

        const [outDepth, outHeight, outWidth] = outputShape
        this.outputShape = outputShape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        bool inBounds(ivec3 coords)
        {
            return  all(greaterThanEqual(coords, minCoords)) && 
                    all(lessThanEqual(coords, maxCoords));
        }

        ivec3 getOutCoords()
        {
            ivec3 cCoords = getOutputCoords();
            return ivec3(cCoords.z, cCoords.y, cCoords.x);
        }

        vec4 getAAt(ivec3 cCoords)
        {
            if (inBounds(cCoords))
                return getA(cCoords.z, cCoords.y, cCoords.x, 0, 0);
            else 
                return vec4(0.0);
        }

        vec4 getBAt(ivec3 cCoords)
        {
            if (inBounds(cCoords))
                return getB(cCoords.z, cCoords.y, cCoords.x, 0, 0);
            else 
                return vec4(1.0);
        }

        vec3 getMaxValues(ivec3 coords)
        {
            vec4 b111 = getBAt(coords + ivec3(${transformCellOffset(-0,-0,-0)}));

            return vec3(b111.x, b111.y, b111.z);
        }

        vec3 getMinValues(ivec3 coords)
        {
            vec4 a011 = getAAt(coords + ivec3(${transformCellOffset(-1,-0,-0)}));
            vec4 a101 = getAAt(coords + ivec3(${transformCellOffset(-0,-1,-0)}));
            vec4 a110 = getAAt(coords + ivec3(${transformCellOffset(-0,-0,-1)}));

            return vec3(a011.x, a101.y, a110.z);
        }

        bool isShadowed(vec3 minValues, vec3 maxValues)
        {
            return all(lessThan(maxValues - minValues, vec3(tolerance)));
        }
                
        void main()
        {
            ivec3 coords = getOutCoords();

            vec3 maxValues = getMaxValues(coords);
            vec3 minValues = getMinValues(coords);

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
        this.outputShape = outputShape
        this.userCode = `
        uvec4 toUint(vec4 v) 
        { 
            return uvec4(round(v)) & 1u; 
        }

        ivec3 getOutCoords()
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getAAt(ivec3 coords)
        {
            return getA(coords.z, coords.y, coords.x);
        }

        vec4 getBAt(ivec3 coords)
        {
            return getB(coords.z, coords.y, coords.x);
        }
                
        void main()
        {
            ivec3 coords = getOutCoords();

            uvec4 sA = toUint(getAAt(coords));
            uvec4 sB = toUint(getBAt(coords));

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
        this.outputShape = outputShape
        this.userCode = `
        uvec4 toUint(vec4 v) 
        { 
            return uvec4(round(v)) & 1u; 
        }

        ivec3 getOutCoords()
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getAAt(ivec3 coords)
        {
            return getA(coords.z, coords.y, coords.x);
        }

        vec4 getBAt(ivec3 coords)
        {
            return getB(coords.z, coords.y, coords.x);
        }

        vec4 getCAt(ivec3 coords)
        {
            return getC(coords.z, coords.y, coords.x);
        }

        vec4 getDAt(ivec3 coords)
        {
            return getD(coords.z, coords.y, coords.x);
        }

        uvec4 bitpack(uvec4 sA, uvec4 sB, uvec4 sC, uvec4 sD) 
        { 
            return (sA << 0) | (sB << 1) | (sC << 2) | (sD << 3);
        }
 
        void main()
        {
            ivec3 coords = getOutCoords();

            uvec4 sA = toUint(getAAt(coords));
            uvec4 sB = toUint(getBAt(coords));
            uvec4 sC = toUint(getCAt(coords));
            uvec4 sD = toUint(getDAt(coords));

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
        this.outputShape = outputShape
        this.userCode = `
        uvec4 toUint(vec4 v)
        { 
            return uvec4(round(v)) & 15u; 
        }

        ivec3 getOutCoords()
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getAAt(ivec3 coords)
        {
            return getA(coords.z, coords.y, coords.x);
        }

        vec4 getBAt(ivec3 coords)
        {
            return getB(coords.z, coords.y, coords.x);
        }

        vec4 getCAt(ivec3 coords)
        {
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

            uvec4 sA = toUint(getAAt(coords));
            uvec4 sB = toUint(getBAt(coords));
            uvec4 sC = toUint(getCAt(coords));

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
        this.outputShape = outputShape
        this.userCode = `
        uvec4 toUint(vec4 v) 
        { 
            return uvec4(round(v)) & 15u; 
        }

        ivec3 getOutCoords() 
        {
            ivec3 c = getOutputCoords();
            return ivec3(c.z, c.y, c.x);
        }

        vec4 getAAt(ivec3 coords) 
        {
            return getA(coords.z, coords.y, coords.x);
        }

        void main() 
        {
            ivec3 coords = getOutCoords();

            uvec4 u = toUint(getAAt(coords));
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
        this.outputShape = outputShape
        this.userCode = `
        ivec4 toInt(vec4 v) 
        { 
            return clamp(ivec4(round(v)), -2048, 2047); 
        }

        ivec3 getOutCoords() 
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getAAt(ivec3 coords) 
        {
            return getA(coords.z, coords.y, coords.x);
        }

        void main() 
        {
            ivec3 coords = getOutCoords();

            ivec4 v = toInt(getAAt(coords)); // -2048..2047 half float precision 
            uvec4 u = uvec4(v + ivec4(2048)); // 0..4095
            uvec4 s = (u >> map) & 1u;

            setOutput(vec4(s));
        }
        `
    }
}

function unidirectionalMinimaMapHollow(
    volume: tf.Tensor3D, 
    holes: tf.Tensor3D,
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
): tf.Tensor5D
{
    const axis = permute[0]
    const backwards = reverse.includes(axis)

    const program = new UnidirectionalMinimaMapHollow(volume.shape, permute, reverse)
    let minima = runWebGLProgram(program, [volume, holes], 'float32', [], true) as tf.Tensor5D
    if (verbose) logTensor('minimaStart', minima)

    const slices = unstackPacked(minima, axis) 
    tf.dispose(minima)

    const shape = slices[0].shape as [number, number, number, 2, 2]
    const propagate = new PropagateUnidirectionalMinimaSlices(shape, permute, reverse)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step) 
    {
        const slice = runWebGLProgram(propagate, [slices[i], slices[i-step]], 'float32', [[i]], true)

        tf.dispose(slices[i])
        slices[i] = slice
    }

    minima = stackPacked(slices, axis) as tf.Tensor5D 
    tf.dispose(slices)
    if (verbose) logTensor('minimaPropagated', minima)

    return minima
}

function unidirectionalMinimaMap(
    volume: tf.Tensor3D, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
): tf.Tensor5D
{
    const axis = permute[0]
    const backwards = reverse.includes(axis)

    const program = new UnidirectionalMinimaMap(volume.shape, permute, reverse)
    let minima = runWebGLProgram(program, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logTensor('minimaStart', minima)

    const slices = unstackPacked(minima, axis) 
    tf.dispose(minima)

    const shape = slices[0].shape as [number, number, number, 2, 2]
    const propagate = new PropagateUnidirectionalMinimaSlices(shape, permute, reverse)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step) 
    {
        const slice = runWebGLProgram(propagate, [slices[i], slices[i-step]], 'float32', [[i]], true)

        tf.dispose(slices[i])
        slices[i] = slice
    }

    minima = stackPacked(slices, axis) as tf.Tensor5D 
    tf.dispose(slices)
    if (verbose) logTensor('minimaPropagated', minima)

    return minima
}

function unidirectionalMaximaMap(
    volume: tf.Tensor3D, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
) : tf.Tensor5D
{
    const program = new UnidirectionalMaximaMap(volume.shape, permute, reverse)
    const maxima = runWebGLProgram(program, [volume], 'float32', [], true) 
    if (verbose) logTensor('maxima', maxima)

    return maxima as tf.Tensor5D
}

function unidirectionalShadowMap(
    minima: tf.Tensor5D, 
    maxima: tf.Tensor5D, 
    permute: Permute, 
    reverse: Reverse, 
    tolerance: number = 0.01,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = minima.shape.slice(0,3) as [number, number, number]
    const program = new UnidirectionalShadowMap(shape, permute, reverse)
    const shadows = runWebGLProgram(program, [minima, maxima], 'float32', [[tolerance]], true)
    if (verbose) logTensor('shadows', shadows)

    return shadows as tf.Tensor3D
}

function bidirectionalShadowMap(
    forwardShadows: tf.Tensor3D, 
    backwardShadows: tf.Tensor3D, 
    verbose: boolean = false
): tf.Tensor3D
{
    const program = new BidirectionalShadowMap(forwardShadows.shape)
    const shadows = runWebGLProgram(program, [forwardShadows, backwardShadows], 'float32', [], true) 
    if (verbose) logTensor('bidirectionalShadows', shadows)

    return shadows as tf.Tensor3D
}

function anisotropicBidirectionalShadowMap(
    shadowMaps: Array4<tf.Tensor3D>, 
    verbose: boolean = false
): tf.Tensor3D
{
    const program = new AnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadows = runWebGLProgram(program, shadowMaps, 'float32', [], true) 
    if (verbose) logTensor('bidirectionalShadows', shadows)

    return shadows as tf.Tensor3D
}

function extendedAnisotropicBidirectionalShadowMap(
    shadowMaps: Array3<tf.Tensor3D>, 
    verbose: boolean = false
): tf.Tensor3D
{
    const program = new ExtendedAnisotropicBidirectionalShadowMap(shadowMaps[0].shape)
    const shadows = runWebGLProgram(program, shadowMaps, 'float32', [], true) 
    if (verbose) logTensor('bidirectionalShadows', shadows)

    return shadows as tf.Tensor3D
}

export function computeUnidirectionalShadowMap(
    volume: tf.Tensor3D, 
    dominantAxis: Axis, 
    octant: Octant, 
    tolerance: number = 0.01,
    verbose: boolean = false
) : tf.Tensor3D
{
    const { permute, reverse } = permuteReverseFromDominantAxisOctant(dominantAxis, octant)

    const minimaMap = unidirectionalMinimaMap(volume, permute, reverse)
    if (verbose) logTensor('minimaMap', minimaMap)

    const maximaMap = unidirectionalMaximaMap(volume, permute, reverse)
    if (verbose) logTensor('maximaMap', maximaMap)

    const shadowsMap = unidirectionalShadowMap(minimaMap, maximaMap, permute, reverse, tolerance)
    if (verbose) logTensor('shadowsMap', shadowsMap)

    tf.dispose([minimaMap, maximaMap])

    return shadowsMap as tf.Tensor3D
}

export function computeBidirectionalShadowMap(
    volume: tf.Tensor3D, 
    dominantAxis: Axis, 
    octant: Octant, 
    tolerance: number = 0.01,
    verbose: boolean = false
) : tf.Tensor3D
{
    const { permute, reverse } = permuteReverseFromDominantAxisOctant(dominantAxis, octant)

    const backwardReverse = complementReverse(reverse)

    const forwardShadowMap = computeUnidirectionalShadowMap(volume, dominantAxis, octant, tolerance)
    if (verbose) logTensor('forwardShadowMap', forwardShadowMap)

    const backwardMinimaMap = unidirectionalMinimaMapHollow(volume, forwardShadowMap, permute, backwardReverse)
    const backwardMaximaMap = unidirectionalMaximaMap(volume, permute, backwardReverse)

    const backwardShadowMap = unidirectionalShadowMap(backwardMinimaMap, backwardMaximaMap, permute, backwardReverse, tolerance)
    if (verbose) logTensor('backwardShadowMap', backwardShadowMap)

    tf.dispose([backwardMinimaMap, backwardMaximaMap])

    const shadowMap = bidirectionalShadowMap(forwardShadowMap, backwardShadowMap)
    if (verbose) logTensor('shadowMap', shadowMap)

    tf.dispose([forwardShadowMap, backwardShadowMap])

    return shadowMap as tf.Tensor3D
}

export function computeBidirectionalShadowMapReverse(
    volume: tf.Tensor3D, 
    dominantAxis: Axis, 
    octant: Octant, 
    tolerance: number = 0.01,
    verbose: boolean = false
) : tf.Tensor3D
{
    const shadowMap = computeBidirectionalShadowMap(volume, dominantAxis, reverseOctant(octant), tolerance, verbose)
    return shadowMap as tf.Tensor3D
}

export function computeExtendedAnisotropicUnidirectionalShadowMap(
    volume: tf.Tensor3D,
    tolerance: number = 0.01,
    verbose: boolean = false
) : tf.Tensor3D
{
    let anisotropicMaps: tf.Tensor3D[] = []
    let extendedMaps: tf.Tensor3D[] = []

    // dominantAxis = x
    anisotropicMaps = []
    anisotropicMaps.push(computeUnidirectionalShadowMap(volume, 'x', '+++', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMap(volume, 'x', '+-+', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMap(volume, 'x', '++-', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMap(volume, 'x', '+--', tolerance))

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)

    // dominantAxis = y
    anisotropicMaps = []
    anisotropicMaps.push(computeUnidirectionalShadowMap(volume, 'y', '+++', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMap(volume, 'y', '-++', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMap(volume, 'y', '++-', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMap(volume, 'y', '-+-', tolerance))

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)

    // dominantAxis = z
    anisotropicMaps = []
    anisotropicMaps.push(computeUnidirectionalShadowMap(volume, 'z', '+++', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMap(volume, 'z', '+-+', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMap(volume, 'z', '-++', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMap(volume, 'z', '--+', tolerance))

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)

    const shadowMap = extendedAnisotropicBidirectionalShadowMap(extendedMaps as Array3<tf.Tensor3D>)
    if (verbose) logExtendedAnisotropicBidirectionalShadowMaps(shadowMap)

    tf.dispose(extendedMaps)
    return shadowMap as tf.Tensor3D
}

export function computeExtendedAnisotropicBidirectionalShadowMap(
    volume: tf.Tensor3D, 
    tolerance: number = 0.01,
    verbose: boolean = false
) : tf.Tensor3D
{
    let anisotropicMaps = [] 
    let extendedMaps = []

    anisotropicMaps = []
    anisotropicMaps.push(computeBidirectionalShadowMap(volume, 'x', '+++', tolerance))
    anisotropicMaps.push(computeBidirectionalShadowMap(volume, 'x', '+-+', tolerance))
    anisotropicMaps.push(computeBidirectionalShadowMap(volume, 'x', '++-', tolerance))
    anisotropicMaps.push(computeBidirectionalShadowMap(volume, 'x', '+--', tolerance))

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)
   
    anisotropicMaps = []
    anisotropicMaps.push(computeBidirectionalShadowMap(volume, 'y', '+++', tolerance))
    anisotropicMaps.push(computeBidirectionalShadowMap(volume, 'y', '-++', tolerance))
    anisotropicMaps.push(computeBidirectionalShadowMap(volume, 'y', '++-', tolerance))
    anisotropicMaps.push(computeBidirectionalShadowMap(volume, 'y', '-+-', tolerance))

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)

    anisotropicMaps = []
    anisotropicMaps.push(computeBidirectionalShadowMap(volume, 'z', '+++', tolerance))
    anisotropicMaps.push(computeBidirectionalShadowMap(volume, 'z', '+-+', tolerance))
    anisotropicMaps.push(computeBidirectionalShadowMap(volume, 'z', '-++', tolerance))
    anisotropicMaps.push(computeBidirectionalShadowMap(volume, 'z', '--+', tolerance))

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)

    const shadowMap = extendedAnisotropicBidirectionalShadowMap(extendedMaps as Array3<tf.Tensor3D>)
    if (verbose) logExtendedAnisotropicBidirectionalShadowMaps(shadowMap)
        
    tf.dispose(extendedMaps)

    return shadowMap as tf.Tensor3D
}

export function computeExtendedAnisotropicBidirectionalShadowMapSingular(
    volume: tf.Tensor3D, 
    tolerance: number = 0.01,
    verbose: boolean = false
) : tf.Tensor3D
{
    const t = computeUnidirectionalShadowMap(volume, 'z', '+++', tolerance)

    let anisotropicMaps = [] 
    let extendedMaps = []

    anisotropicMaps = []
    anisotropicMaps.push(tf.onesLike(t)) // [2,1,0], [   ]
    anisotropicMaps.push(tf.onesLike(t)) // [2,1,0], [  1]
    anisotropicMaps.push(tf.onesLike(t)) // [2,1,0], [  0]
    anisotropicMaps.push(tf.onesLike(t)) // [2,1,0], [1,0]

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)
   
    anisotropicMaps = []
    anisotropicMaps.push(tf.onesLike(t)) // [1,2,0], [   ]
    anisotropicMaps.push(tf.onesLike(t)) // [1,2,0], [  2]
    anisotropicMaps.push(tf.onesLike(t)) // [1,2,0], [  0]
    anisotropicMaps.push(tf.onesLike(t)) // [1,2,0], [2,0]

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)

    anisotropicMaps = []
    anisotropicMaps.push(tf.clone(t)) // [0,1,2], [   ]
    anisotropicMaps.push(tf.onesLike(t)) // [0,1,2], [  1]
    anisotropicMaps.push(tf.onesLike(t)) // [0,1,2], [  2]
    anisotropicMaps.push(tf.onesLike(t)) // [0,1,2], [1,2]

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)

    const shadowMap = extendedAnisotropicBidirectionalShadowMap(extendedMaps as Array3<tf.Tensor3D>)
    if (verbose) logExtendedAnisotropicBidirectionalShadowMaps(shadowMap)
        
    tf.dispose(extendedMaps)
    tf.dispose(t)

    return shadowMap as tf.Tensor3D
}

// reference functions 

export function computeUnidirectionalShadowMapReference(
    volume: tf.Tensor3D, 
    dominantAxis: Axis, 
    octant: Octant, 
    tolerance: number = 0.01,
    verbose: boolean = false
) : tf.Tensor3D
{
    const { permute, reverse } = permuteReverseFromDominantAxisOctant(dominantAxis, octant)

    const reversed = volume.reverse(reverse) as tf.Tensor3D
    const transposed = reversed.transpose(permute) as tf.Tensor3D
    tf.dispose(reversed)

    const minimaMap = unidirectionalMinimaMap(transposed, [0,1,2], [])
    if (verbose) logTensor('minimaMap', minimaMap)

    const maximaMap = unidirectionalMaximaMap(transposed, [0,1,2], [])
    if (verbose) logTensor('maximaMap', maximaMap)

    const shadowMap = unidirectionalShadowMap(minimaMap, maximaMap, [0,1,2], [], tolerance)
    if (verbose) logTensor('shadowMap', shadowMap)

    tf.dispose([minimaMap, maximaMap])

    tf.dispose(transposed)
    const untransposed = shadowMap.transpose(inversePermutation(permute))
    tf.dispose(shadowMap)
    const unreversed = untransposed.reverse(reverse)
    tf.dispose(untransposed)
    
    return unreversed as tf.Tensor3D
}

export function computeExtendedAnisotropicUnidirectionalShadowMapReference(
    volume: tf.Tensor3D,
    tolerance: number = 0.01,
    verbose: boolean = false
) : tf.Tensor3D
{
    let anisotropicMaps: tf.Tensor3D[] = []
    let extendedMaps: tf.Tensor3D[] = []

    // dominantAxis = x
    anisotropicMaps = []
    anisotropicMaps.push(computeUnidirectionalShadowMapReference(volume, 'x', '+++', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMapReference(volume, 'x', '+-+', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMapReference(volume, 'x', '++-', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMapReference(volume, 'x', '+--', tolerance))

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)

    // dominantAxis = y
    anisotropicMaps = []
    anisotropicMaps.push(computeUnidirectionalShadowMapReference(volume, 'y', '+++', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMapReference(volume, 'y', '-++', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMapReference(volume, 'y', '++-', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMapReference(volume, 'y', '-+-', tolerance))

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)

    // dominantAxis = z
    anisotropicMaps = []
    anisotropicMaps.push(computeUnidirectionalShadowMapReference(volume, 'z', '+++', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMapReference(volume, 'z', '+-+', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMapReference(volume, 'z', '-++', tolerance))
    anisotropicMaps.push(computeUnidirectionalShadowMapReference(volume, 'z', '--+', tolerance))

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)

    const shadowMap = extendedAnisotropicBidirectionalShadowMap(extendedMaps as Array3<tf.Tensor3D>)
    if (verbose) logExtendedAnisotropicBidirectionalShadowMaps(shadowMap)

    tf.dispose(extendedMaps)
    return shadowMap as tf.Tensor3D
}

// helper functions

function logTensor(name: string, tensor: tf.Tensor)
{
    console.log(name, tf.tidy(() => tensor.mean([0,1,2]).dataSync())) 
}

function logAnisotropicBidirectionalShadowMaps(shadowMaps: tf.Tensor3D)
{
    const unpack = new UnpackAnisotropicBidirectionalShadowMap(shadowMaps.shape)

    console.log('shadowMap0', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[0]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMap1', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[1]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMap2', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[2]]).mean([0,1,2]).dataSync())) 
    console.log('shadowMap3', tf.tidy(() => runWebGLProgram(unpack, [shadowMaps], 'float32', [[3]]).mean([0,1,2]).dataSync())) 
}

function logExtendedAnisotropicBidirectionalShadowMaps(shadowMaps: tf.Tensor3D)
{
    const unpack = new UnpackExtendedAnisotropicBidirectionalShadowMap(shadowMaps.shape)

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


