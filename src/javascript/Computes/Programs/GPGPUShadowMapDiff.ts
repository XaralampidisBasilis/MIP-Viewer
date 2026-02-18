import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { unstackPacked } from './unstack_packed_keepDims_webgl'
import { stackPacked } from './stack_packed_keepDims_webgl'

type Axis = 0 | 1 | 2
type Permute = [Axis, Axis, Axis]
type Reverse = Axis[]

class UnidirectionalDifferenceMap implements GPGPUProgram 
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
   
        const transformOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)
            for (const a of reverse) old[a] = - old[a]
            return old.toReversed().join(',')
        }

        const [inDepth, inHeight, inWidth] = volumeShape
        this.outputShape = [inDepth, inHeight, inWidth, 2, 2]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        bool inBounds(ivec3 coords)
        {
            return  
                all(greaterThanEqual(coords, minCoords)) && 
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

        void main()
        {
            ivec3 coords = getOutCoords();

            float v111 = getAAt(coords + ivec3(${transformOffset(-0,-0,-0)}));
            float v110 = getAAt(coords + ivec3(${transformOffset(-0,-0,-1)}));
            float v100 = getAAt(coords + ivec3(${transformOffset(-0,-1,-1)}));
            float v010 = getAAt(coords + ivec3(${transformOffset(-1,-0,-1)}));
            float v000 = getAAt(coords + ivec3(${transformOffset(-1,-1,-1)}));

            float d000 = v000 - v111;
            float d010 = v010 - v111;
            float d100 = v100 - v111;
            float d110 = v110 - v111;

            setOutput(vec4(d000, d010, d100, d110));
        }
        `
    }
}

class PropagateUnidirectionalDifferenceSlices implements GPGPUProgram 
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

        const transformOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)
            for (const a of reverse) old[a] = - old[a]

            const axis = permute[0]
            old[axis] = 0
            
            return old.toReversed().join(',')
        }

        const [outDepth, outHeight, outWidth] = outputShape.slice(0,3)
        this.outputShape = outputShape 
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        float min4(vec4 v) 
        { 
            return min(min(min(v.x, v.y), v.z), v.w); 
        }

        bool inBounds(ivec3 coords)
        {
            return 
                all(greaterThanEqual(coords, minCoords)) && 
                all(lessThanEqual(coords, maxCoords));
        }

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getAAt(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x, 0, 0);
        }

        vec4 getBAt(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getB(coords.z, coords.y, coords.x, 0, 0); 
        }

        void main()
        {
            ivec3 coords = getOutCoords();

            vec4 d111 = getAAt(coords + ivec3(${transformOffset(-0,-0,-0)}));
            vec4 d110 = getBAt(coords + ivec3(${transformOffset(-0,-0,-1)}));
            vec4 d100 = getBAt(coords + ivec3(${transformOffset(-0,-1,-1)}));
            vec4 d010 = getBAt(coords + ivec3(${transformOffset(-1,-0,-1)}));
            vec4 d000 = getBAt(coords + ivec3(${transformOffset(-1,-1,-1)}));

            d111.x += max(min4(d000), 0.0);
            d111.y += max(min4(d010), 0.0);
            d111.z += max(min4(d100), 0.0);
            d111.w += max(min4(d110), 0.0);

            setOutput(d111);
        }
        `
    }
}

class PropagateGatedUnidirectionalDifferenceSlices implements GPGPUProgram 
{
    variableNames = ['A', 'B', 'G']
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
      
        const transformOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)

            for (const a of reverse) old[a] = - old[a]

            const axis = permute[0]
            old[axis] = 0
            
            return old.toReversed().join(',')
        }

        const substituteSlice = (cx: string, cy: string, cz: string): string => 
        {
            const coords = [cz, cy, cx]

            const axis = permute[0]
            coords[axis] = 'slice'
            
            return coords.toReversed().join(',')
        }

        const [outDepth, outHeight, outWidth] = outputShape.slice(0,3)
        this.outputShape = outputShape 
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        float min4(vec4 v) 
        { 
            return min(min(min(v.x, v.y), v.z), v.w); 
        }

        bool inBounds(ivec3 coords)
        {
            return 
                all(greaterThanEqual(coords, minCoords)) && 
                all(lessThanEqual(coords, maxCoords));
        }

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getAAt(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x, 0, 0);
        }

        vec4 getBAt(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getB(coords.z, coords.y, coords.x, 0, 0);
        }

        vec4 getGAt(ivec3 coords)
        {
            coords = ivec3(${substituteSlice('coords.x','coords.y','coords.z')});
            return getG(coords.z, coords.y, coords.x, 0, 0);
        }

        void main()
        {
            ivec3 coords = getOutCoords();
            bvec4 gates = greaterThan(getGAt(coords), vec4(0.5));

            vec4 d111 = getAAt(coords + ivec3(${transformOffset(-0,-0,-0)}));
            vec4 d110 = getBAt(coords + ivec3(${transformOffset(-0,-0,-1)}));
            vec4 d100 = getBAt(coords + ivec3(${transformOffset(-0,-1,-1)}));
            vec4 d010 = getBAt(coords + ivec3(${transformOffset(-1,-0,-1)}));
            vec4 d000 = getBAt(coords + ivec3(${transformOffset(-1,-1,-1)}));

            float m000 = min4(d000);
            float m010 = min4(d010);
            float m100 = min4(d100);
            float m110 = min4(d110);

            d111.x += (gates.x) ? max(m000, 0.0) : m000;
            d111.y += (gates.y) ? max(m010, 0.0) : m010;
            d111.z += (gates.z) ? max(m100, 0.0) : m100;
            d111.w += (gates.w) ? max(m110, 0.0) : m110;

            setOutput(d111);
        }
        `
    }
}

class UnidirectionalGateMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    customUniforms = [{ name: 'tolerance', type: 'float' as const }]

    constructor(
        outputShape: [number, number, number, 2, 2], 
        permute: Permute = [0,1,2], 
        reverse: Reverse = []
    ) {
    
        const transformOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)
            for (const a of reverse) old[a] = - old[a]
            return old.toReversed().join(',')
        }

        const [inDepth, inHeight, inWidth] = outputShape
        this.outputShape = outputShape
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        bool inBounds(ivec3 coords)
        {
            return 
                all(greaterThanEqual(coords, minCoords)) && 
                all(lessThanEqual(coords, maxCoords));
        }

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getAAt(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x, 0, 0);
        }

        void main()
        {
            ivec3 coords = getOutCoords();

            vec4 d001 = getAAt(coords + ivec3(${transformOffset(0,0,1)}));
            vec4 d011 = getAAt(coords + ivec3(${transformOffset(0,1,1)}));
            vec4 d101 = getAAt(coords + ivec3(${transformOffset(1,0,1)}));
            vec4 d111 = getAAt(coords + ivec3(${transformOffset(1,1,1)}));

            bool g111 = d111.x > -tolerance;
            bool g101 = d101.y > -tolerance;
            bool g011 = d011.z > -tolerance;
            bool g001 = d001.w > -tolerance;

            setOutput(vec4(g001, g011, g101, g111));

            // bool g001 = all(greaterThan(d001, vec4(-tolerance)));
            // bool g011 = all(greaterThan(d011, vec4(-tolerance)));
            // bool g101 = all(greaterThan(d101, vec4(-tolerance)));
            // bool g111 = all(greaterThan(d111, vec4(-tolerance)));

            // setOutput(vec4(g001, g011, g101, g111));
        }
        `
    }
}

class UnidirectionalGateMap2 implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = false

    constructor(inputShape: [number, number, number]) 
    {
        this.outputShape = inputShape
        this.userCode = `

        ivec3 getOutCoords()
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getA(ivec3 coords)
        {
            return getA(coords.z, coords.y, coords.x, 0, 0);
        }

        void main()
        {
            ivec3 coords = getOutCoords();
            vec4 d = getA(coords);
            bool b = all(greaterThanEqual(d, vec4(0.0)));

            setOutput(float(b));
        }
        `
    }
}

class UnidirectionalShadowMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = false

    customUniforms = [{ name: 'tolerance', type: 'float' as const }]

    constructor(
        volumeShape: [number, number, number], 
        permute: Permute = [0,1,2], 
        reverse: Reverse = []
    ) {
        
        const transformOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)
            for (const a of reverse) old[a] = 1 - old[a]
            return old.toReversed().join(',')
        }

        const [inDepth, inHeight, inWidth] = volumeShape
        const [outDepth, outHeight, outWidth] = volumeShape.map(x => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth]  
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        struct CellValues 
        { 
            vec4 d000; 
            vec4 d100; 
            vec4 d010; 
            vec4 d001; 
            vec4 d011; 
            vec4 d101; 
            vec4 d110; 
            vec4 d111; 
        }; 

        bool inBounds(ivec3 coords)
        {
            return  
                all(greaterThanEqual(coords, minCoords)) && 
                all(lessThanEqual(coords, maxCoords));
        }

        ivec3 getOutCoords()
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getAAt(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            
            if (inBounds(coords)) 
                return getA(coords.z, coords.y, coords.x, 0, 0);
            else
                return vec4(0.0);
        }

        CellValues getValues(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            CellValues c;
            c.d000 = getAAt(voxelCoords + ivec3(${transformOffset(0,0,0)}));
            c.d010 = getAAt(voxelCoords + ivec3(${transformOffset(0,1,0)}));
            c.d100 = getAAt(voxelCoords + ivec3(${transformOffset(1,0,0)}));
            c.d110 = getAAt(voxelCoords + ivec3(${transformOffset(1,1,0)}));
            c.d001 = getAAt(voxelCoords + ivec3(${transformOffset(0,0,1)}));
            c.d011 = getAAt(voxelCoords + ivec3(${transformOffset(0,1,1)}));
            c.d101 = getAAt(voxelCoords + ivec3(${transformOffset(1,0,1)}));
            c.d111 = getAAt(voxelCoords + ivec3(${transformOffset(1,1,1)}));

            return c;
        }

        bool isShadowed(CellValues c)
        {            
            return  
                all(greaterThan(c.d111, vec4(-tolerance))) &&
                all(greaterThan(c.d101, vec4(-tolerance))) &&
                all(greaterThan(c.d011, vec4(-tolerance))) &&
                all(greaterThan(c.d001, vec4(-tolerance)));
        }

        // bool isShadowed2(CellValues c)
        // {
        //     float m = 0.0;

        //     m = min(m, c.d111.x); 
        //     m = min(m, c.d111.y); 
        //     m = min(m, c.d111.z); 
        //     m = min(m, c.d111.w); 
        //     m = min(m, c.d101.y); 
        //     m = min(m, c.d101.w); 
        //     m = min(m, c.d011.z); 
        //     m = min(m, c.d011.w); 
        //     m = min(m, c.d001.w); 

        //     return (m > -tolerance);
        // }

        void main()
        {
            ivec3 coords = getOutCoords();
            CellValues cellValues = getValues(coords);
        
            setOutput(float(isShadowed(cellValues)));
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

function propagateGatedUnidirectionalDifferenceSlices(
    differences: tf.Tensor5D, 
    gates: tf.Tensor3D,
    permute: Permute, 
    reverse: Reverse
): tf.Tensor5D
{
    const axis = permute[0]
    const backwards = reverse.includes(axis)

    const slices = unstackPacked(differences, axis) 
    differences.dispose()

    const shape = slices[0].shape as [number, number, number, 2, 2]
    const program = new PropagateGatedUnidirectionalDifferenceSlices(shape, permute, reverse)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step) 
    {
        const slice = runWebGLProgram(program, [slices[i], slices[i-step], gates], 'float32', [[i]], true)

        tf.dispose(slices[i])
        slices[i] = slice
    }

    differences = stackPacked(slices, axis) as tf.Tensor5D 
    tf.dispose(slices)

    return differences
}

function propagatedGatedUnidirectionalDifferenceMap(
    volume: tf.Tensor3D, 
    gates: tf.Tensor3D,
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
): tf.Tensor5D
{
    const program = new UnidirectionalDifferenceMap(volume.shape, permute, reverse)
    let differences = runWebGLProgram(program, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logTensor('differencesStart', differences)

    differences = propagateGatedUnidirectionalDifferenceSlices(differences, gates, permute, reverse) 
    if (verbose) logTensor('differencesPropagated', differences)

    return differences as tf.Tensor5D
}

function propagateUnidirectionalDifferenceSlices(
    differences: tf.Tensor5D, 
    permute: Permute, 
    reverse: Reverse
): tf.Tensor5D
{
    const axis = permute[0]
    const backwards = reverse.includes(axis)

    const slices = unstackPacked(differences, axis) 
    tf.dispose(differences)

    const shape = slices[0].shape as [number, number, number, 2, 2]
    const program = new PropagateUnidirectionalDifferenceSlices(shape, permute, reverse)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step) 
    {
        const slice = runWebGLProgram(program, [slices[i], slices[i-step]], 'float32', [[i]], true)

        tf.dispose(slices[i])
        slices[i] = slice
    }

    differences = stackPacked(slices, axis) as tf.Tensor5D 
    tf.dispose(slices)

    return differences
}

function unidirectionalDifferenceMap(
    volume: tf.Tensor3D, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
): tf.Tensor5D
{
    const program = new UnidirectionalDifferenceMap(volume.shape, permute, reverse)
    let differences = runWebGLProgram(program, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logTensor('differencesStart', differences)

    differences = propagateUnidirectionalDifferenceSlices(differences, permute, reverse)
    if (verbose) logTensor('differencesPropagated', differences)

    return differences  as tf.Tensor5D
}

function unidirectionalGateMap(
    differences: tf.Tensor5D, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = differences.shape as [number, number, number, 2, 2]
    const program = new UnidirectionalGateMap(shape, permute, reverse)
    // const program = new UnidirectionalGateMap2(shape)

    const shadows = runWebGLProgram(program, [differences], 'float32', [[0.005]], true) 
    if (verbose) logTensor('shadows', shadows)

    return shadows as tf.Tensor3D
}

function unidirectionalShadowMap(
    differences: tf.Tensor5D, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = differences.shape.slice(0,3) as [number, number, number]
    const program = new UnidirectionalShadowMap(shape, permute, reverse)

    const shadows = runWebGLProgram(program, [differences], 'float32', [[0.005]], true)
    tf.dispose(differences)

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

    tf.dispose([forwardShadows, backwardShadows])

    return shadows as tf.Tensor3D
}

export function computeUnidirectionalShadowMap(
    volume: tf.Tensor3D, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
) : tf.Tensor3D
{
    const differences = unidirectionalDifferenceMap(volume, permute, reverse, verbose)
    const shadows = unidirectionalShadowMap(differences, permute, reverse, verbose)

    return shadows as tf.Tensor3D
}

export function computeBidirectionalShadowMap(
    volume: tf.Tensor3D, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
) : tf.Tensor3D
{
    const posShadows = computeUnidirectionalShadowMap(volume, permute, reverse)
    if (verbose) logTensor('posShadows', posShadows)

    const negShadows = computeUnidirectionalShadowMap(volume, permute, complementReverse(reverse))
    if (verbose) logTensor('negShadows', negShadows)
        
    const or = new BidirectionalShadowMap(posShadows.shape)
    const shadows = runWebGLProgram(or, [posShadows, negShadows], 'float32', [], true) as tf.Tensor3D
    tf.dispose([posShadows, negShadows])
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
        computeUnidirectionalShadowMap(volume, permute, reverseA),
        computeUnidirectionalShadowMap(volume, permute, reverseB),
        computeUnidirectionalShadowMap(volume, permute, reverseC),
        computeUnidirectionalShadowMap(volume, permute, reverseD),
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

export function computeBidirectionalShadowMapDebug(
    volume: tf.Tensor3D, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
) : tf.Tensor3D
{
    const forwardDifferences = unidirectionalDifferenceMap(volume, permute, reverse)
    if (verbose) logTensor('forwardDifferences', forwardDifferences)

    const forwardGates = unidirectionalGateMap(forwardDifferences, permute, reverse)
    if (verbose) logTensor('forwardGates', forwardGates)

    const forwardShadows = unidirectionalShadowMap(forwardDifferences, permute, reverse)
    if (verbose) logTensor('forwardShadows', forwardShadows)

    tf.dispose(forwardDifferences)

    const compReverse = complementReverse(reverse)
    const backwardDifferences = propagatedGatedUnidirectionalDifferenceMap(volume, forwardGates, permute, compReverse)
    if (verbose) logTensor('backwardDifferences', backwardDifferences)

    tf.dispose(forwardGates)

    const backwardShadows = unidirectionalShadowMap(backwardDifferences, permute, compReverse)
    if (verbose) logTensor('backwardShadows', backwardShadows)

    tf.dispose(backwardDifferences)

    const bidirectionalShadows = bidirectionalShadowMap(forwardShadows, backwardShadows)
    if (verbose) logTensor('bidirectionalShadows', bidirectionalShadows)

    return bidirectionalShadows as tf.Tensor3D
}

export function computeExtendedAnisotropicBidirectionalShadowMapDebug(volumeMap: tf.Tensor3D, verbose: boolean = false) : tf.Tensor3D
{
    let o1,o2,o3,o4,ox,oy,oz

    // let t = computeUnidirectionalShadowMap(volumeMap, [0,1,2], [])
    // let t = computeUnidirectionalShadowMap(volumeMap, [0,1,2], [0,1,2])
    let t = computeBidirectionalShadowMapDebug(volumeMap,  [0,1,2], [], true)

    o1 = tf.onesLike(t) // computeBidirectionalShadowMap(volumeMap, [2,1,0], [   ])
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

function logAnisotropicBidirectionalShadowMaps(occlusionMaps: tf.Tensor3D)
{
    const unpack = new UnpackAnisotropicBidirectionalShadowMap(occlusionMaps.shape)

    console.log('occlusionMap0', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[0]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMap1', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[1]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMap2', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[2]]).mean([0,1,2]).dataSync())) 
    console.log('occlusionMap3', tf.tidy(() => runWebGLProgram(unpack, [occlusionMaps], 'float32', [[3]]).mean([0,1,2]).dataSync())) 
}

function logExtendedAnisotropicBidirectionalShadowMaps(occlusionMaps: tf.Tensor3D)
{
    const unpack = new UnpackExtendedAnisotropicBidirectionalShadowMap(occlusionMaps.shape)

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
