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
} from '../../Utils/ShadowMapUtils'
import { maxPool3d, minPool3d, avgPool3d } from './pool3d'

type Array3<T> = [T, T, T]
type Array4<T> = [T, T, T, T]

class UnidirectionalDifferences implements GPGPUProgram 
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

class PropagateUnidirectionalDifferences implements GPGPUProgram 
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

class PropagateUnidirectionalDifferencesGates implements GPGPUProgram 
{
    variableNames = ['A', 'B', 'C']
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

        vec4 getCAt(ivec3 coords)
        {
            coords = ivec3(${substituteSlice('coords.x','coords.y','coords.z')});
            return getC(coords.z, coords.y, coords.x, 0, 0);
        }

        void main()
        {
            ivec3 coords = getOutCoords();
            bvec4 gates = greaterThan(getCAt(coords), vec4(0.5));

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

class UnidirectionalGates implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(
        inputShape: [number, number, number], 
        permute: Permute = [0,1,2], 
        reverse: Reverse = []
    ) {
      
        const transformOffset = (ox: number, oy: number, oz: number): string => 
        {
            const old = applyPermutation([oz, oy, ox], permute)
            for (const a of reverse) old[a] = 1 - old[a]
            return old.toReversed().join(',')
        }

        const [inDepth, inHeight, inWidth] = inputShape
        const [outDepth, outHeight, outWidth] = inputShape.map((x) => x - 1)
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]
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
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        void main()
        {
            ivec3 coords = getOutCoords();

            float s110 = getAAt(coords + ivec3(${transformOffset(-0,-0,-1)}));
            float s100 = getAAt(coords + ivec3(${transformOffset(-0,-1,-1)}));
            float s010 = getAAt(coords + ivec3(${transformOffset(-1,-0,-1)}));
            float s000 = getAAt(coords + ivec3(${transformOffset(-1,-1,-1)}));
            float s111 = getAAt(coords + ivec3(${transformOffset(-0,-0,-0)}));
            float s101 = getAAt(coords + ivec3(${transformOffset(-0,-1,-0)}));
            float s011 = getAAt(coords + ivec3(${transformOffset(-1,-0,-0)}));
            float s001 = getAAt(coords + ivec3(${transformOffset(-1,-1,-0)}));

            float g110 = 1.0 - s110;
            float g100 = 1.0 - s100;
            float g010 = 1.0 - s010;
            float g000 = 1.0 - s000;

            setOutput(vec4(g000, g010, g100, g110));
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
        const [outDepth, outHeight, outWidth] = volumeShape.map((x) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth]
        this.userCode = `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        struct CellDifferences 
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
            return getA(coords.z, coords.y, coords.x, 0, 0);
        }

        CellDifferences getDifferences(ivec3 coords)
        {
            coords = coords - 1;

            CellDifferences c;
            c.d111 = getAAt(coords + ivec3(${transformOffset(1,1,1)}));
            c.d101 = getAAt(coords + ivec3(${transformOffset(1,0,1)}));
            c.d011 = getAAt(coords + ivec3(${transformOffset(0,1,1)}));
            c.d001 = getAAt(coords + ivec3(${transformOffset(0,0,1)}));
            c.d110 = getAAt(coords + ivec3(${transformOffset(1,1,0)}));
            c.d100 = getAAt(coords + ivec3(${transformOffset(1,0,0)}));
            c.d010 = getAAt(coords + ivec3(${transformOffset(0,1,0)}));
            c.d000 = getAAt(coords + ivec3(${transformOffset(0,0,0)}));
        
            return c;
        }

        bool isShadowed(CellDifferences c)
        {            
            return  
                all(greaterThan(c.d111 + tolerance, vec4(0))) &&
                all(greaterThan(c.d101 + tolerance, vec4(0))) &&
                all(greaterThan(c.d011 + tolerance, vec4(0))) &&
                all(greaterThan(c.d001 + tolerance, vec4(0)));
        }

        void main()
        {
            ivec3 coords = getOutCoords();
            CellDifferences c = getDifferences(coords);
        
            setOutput(float(isShadowed(c)));
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

    constructor(outputShape: [number, number, number]) 
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

// abstract functions

function unidirectionalDifferencesGates(
    volume: tf.Tensor3D, 
    gates: tf.Tensor5D,
    permute: Permute, 
    reverse: Reverse,
    verbose: boolean = false
): tf.Tensor5D
{
    const axis = permute[0]
    const backwards = reverse.includes(axis)

    const program = new UnidirectionalDifferences(volume.shape, permute, reverse)
    let differences = runWebGLProgram(program, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logTensor('differencesStart', differences)

    const slices = unstackPacked(differences, axis) 
    tf.dispose(differences)

    const shape = slices[0].shape as [number, number, number, 2, 2]
    const propagate = new PropagateUnidirectionalDifferencesGates(shape, permute, reverse)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step) 
    {
        const slice = runWebGLProgram(propagate, [slices[i], slices[i-step], gates], 'float32', [[i]], true)

        tf.dispose(slices[i])
        slices[i] = slice
    }

    differences = stackPacked(slices, axis) as tf.Tensor5D 
    tf.dispose(slices)
    if (verbose) logTensor('differencesPropagated', differences)

    return differences
}

function unidirectionalDifferences(
    volume: tf.Tensor3D, 
    permute: Permute, 
    reverse: Reverse,
    verbose: boolean = false
): tf.Tensor5D
{
    const axis = permute[0]
    const backwards = reverse.includes(axis)

    const program = new UnidirectionalDifferences(volume.shape, permute, reverse)
    let differences = runWebGLProgram(program, [volume], 'float32', [], true) as tf.Tensor5D
    if (verbose) logTensor('differencesStart', differences)

    const slices = unstackPacked(differences, axis) 
    tf.dispose(differences)

    const shape = slices[0].shape as [number, number, number, 2, 2]
    const propagate = new PropagateUnidirectionalDifferences(shape, permute, reverse)

    const start = backwards ? slices.length - 2 : 1
    const end = backwards ? -1 : slices.length
    const step = backwards ? -1 : 1

    for (let i = start; i !== end; i += step) 
    {
        const slice = runWebGLProgram(propagate, [slices[i], slices[i-step]], 'float32', [[i]], true)

        tf.dispose(slices[i])
        slices[i] = slice
    }

    differences = stackPacked(slices, axis) as tf.Tensor5D 
    tf.dispose(slices)
    if (verbose) logTensor('differencesPropagated', differences)

    return differences
}

function unidirectionalGates(
    shadowMap: tf.Tensor3D, 
    permute: Permute, 
    reverse: Reverse, 
    verbose: boolean = false
): tf.Tensor5D
{
    const shape = shadowMap.shape as [number, number, number]
    const program = new UnidirectionalGates(shape, permute, reverse)

    const gates = runWebGLProgram(program, [shadowMap], 'float32', [], true) 
    if (verbose) logTensor('gates', gates)

    return gates as tf.Tensor5D
}

function unidirectionalShadowMap(
    differences: tf.Tensor5D, 
    permute: Permute, 
    reverse: Reverse, 
    tolerance: number = 0.01,
    verbose: boolean = false
): tf.Tensor3D
{
    const shape = differences.shape.slice(0,3) as [number, number, number]
    const program = new UnidirectionalShadowMap(shape, permute, reverse)

    const shadowMap = runWebGLProgram(program, [differences], 'float32', [[tolerance]], true)
    if (verbose) logTensor('shadowMap', shadowMap)

    return shadowMap as tf.Tensor3D
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

// compute functions 

export function computeUnidirectionalShadowMap(
    volume: tf.Tensor3D, 
    dominantAxis: Axis, 
    octant: Octant, 
    tolerance: number = 0.01,
    verbose: boolean = false
) : tf.Tensor3D
{
    const { permute, reverse } = permuteReverseFromDominantAxisOctant(dominantAxis, octant)

    const differences = unidirectionalDifferences(volume, permute, reverse) as tf.Tensor5D
    if (verbose) logTensor('differencesPropagated', differences)

    const shadowMap = unidirectionalShadowMap(differences, permute, reverse, tolerance) as tf.Tensor3D
    if (verbose) logTensor('shadowMap', shadowMap)

    tf.dispose(differences)

    return shadowMap as tf.Tensor3D
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

    const forwardShadowMap = computeUnidirectionalShadowMap(volume, dominantAxis, octant, tolerance)
    if (verbose) logTensor('forwardShadowMap', forwardShadowMap)

    const backwardReverse = complementReverse(reverse)
    const backwardGates = unidirectionalGates(forwardShadowMap, permute, backwardReverse)

    const backwardDifferences = unidirectionalDifferencesGates(volume, backwardGates, permute, backwardReverse)
    tf.dispose(backwardGates)

    const backwardShadowMap = unidirectionalShadowMap(backwardDifferences, permute, backwardReverse, tolerance)
    if (verbose) logTensor('backwardShadowMap', backwardShadowMap)
    tf.dispose(backwardDifferences)

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

export function computeBidirectionalBlockShadowMap(
    volume: tf.Tensor3D, 
    dominantAxis: Axis, 
    octant: Octant, 
    tolerance: number = 0.01,
    blockSize: number = 2,
    verbose: boolean = false
) : tf.Tensor3D
{
    const { permute, reverse } = permuteReverseFromDominantAxisOctant(dominantAxis, octant)

    const forwardShadowMap = computeUnidirectionalShadowMap(volume, dominantAxis, octant, tolerance)
    if (verbose) logTensor('forwardShadowMap', forwardShadowMap)

    const backwardReverse = complementReverse(reverse)
    const backwardGates = unidirectionalGates(forwardShadowMap, permute, backwardReverse)

    const backwardDifferences = unidirectionalDifferencesGates(volume, backwardGates, permute, backwardReverse)
    tf.dispose(backwardGates)

    const backwardShadowMap = unidirectionalShadowMap(backwardDifferences, permute, backwardReverse, tolerance)
    if (verbose) logTensor('backwardShadowMap', backwardShadowMap)
    tf.dispose(backwardDifferences)

    const shadowMap = bidirectionalShadowMap(forwardShadowMap, backwardShadowMap)
    if (verbose) logTensor('bidirectionalShadowMap', shadowMap)
    tf.dispose([forwardShadowMap, backwardShadowMap])

    const blockShadowMap = minPool3d(shadowMap, blockSize, blockSize, 1, 'ceil')
    if (verbose) logTensor('blockShadowMap', blockShadowMap)
    tf.dispose(shadowMap)

    return blockShadowMap as tf.Tensor3D
}

export function computeExtendedAnisotropicBidirectionalBlockShadowMap(
    volume: tf.Tensor3D, 
    tolerance: number = 0.01,
    blockSize: number = 2,
    verbose: boolean = false
) : tf.Tensor3D
{
    let anisotropicMaps = [] 
    let extendedMaps = []

    anisotropicMaps = []
    anisotropicMaps.push(computeBidirectionalBlockShadowMap(volume, 'x', '+++', tolerance, blockSize))
    anisotropicMaps.push(computeBidirectionalBlockShadowMap(volume, 'x', '+-+', tolerance, blockSize))
    anisotropicMaps.push(computeBidirectionalBlockShadowMap(volume, 'x', '++-', tolerance, blockSize))
    anisotropicMaps.push(computeBidirectionalBlockShadowMap(volume, 'x', '+--', tolerance, blockSize))

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)
   
    anisotropicMaps = []
    anisotropicMaps.push(computeBidirectionalBlockShadowMap(volume, 'y', '+++', tolerance, blockSize))
    anisotropicMaps.push(computeBidirectionalBlockShadowMap(volume, 'y', '-++', tolerance, blockSize))
    anisotropicMaps.push(computeBidirectionalBlockShadowMap(volume, 'y', '++-', tolerance, blockSize))
    anisotropicMaps.push(computeBidirectionalBlockShadowMap(volume, 'y', '-+-', tolerance, blockSize))

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)

    anisotropicMaps = []
    anisotropicMaps.push(computeBidirectionalBlockShadowMap(volume, 'z', '+++', tolerance, blockSize))
    anisotropicMaps.push(computeBidirectionalBlockShadowMap(volume, 'z', '+-+', tolerance, blockSize))
    anisotropicMaps.push(computeBidirectionalBlockShadowMap(volume, 'z', '-++', tolerance, blockSize))
    anisotropicMaps.push(computeBidirectionalBlockShadowMap(volume, 'z', '--+', tolerance, blockSize))

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)

    const shadowMap = extendedAnisotropicBidirectionalShadowMap(extendedMaps as Array3<tf.Tensor3D>)
    if (verbose) logExtendedAnisotropicBidirectionalShadowMaps(shadowMap)
        
    tf.dispose(extendedMaps)

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

    const differences = unidirectionalDifferences(transposed, [0,1,2], []) as tf.Tensor5D
    if (verbose) logTensor('differencesPropagated', differences)

    const shadowMap = unidirectionalShadowMap(differences, [0,1,2], [], tolerance) as tf.Tensor3D
    if (verbose) logTensor('shadowMap', shadowMap)

    tf.dispose(differences)

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

// debug functions 

export function computeExtendedAnisotropicBidirectionalShadowMapSingular(
    volume: tf.Tensor3D, 
    tolerance: number = 0.01,
    verbose: boolean = false
) : tf.Tensor3D
{
    const tempMap = computeUnidirectionalShadowMap(volume, 'z', '+++', tolerance)

    let anisotropicMaps = [] 
    let extendedMaps = []

    anisotropicMaps = []
    anisotropicMaps.push(tf.onesLike(tempMap)) // [2,1,0], [   ]
    anisotropicMaps.push(tf.onesLike(tempMap)) // [2,1,0], [  1]
    anisotropicMaps.push(tf.onesLike(tempMap)) // [2,1,0], [  0]
    anisotropicMaps.push(tf.onesLike(tempMap)) // [2,1,0], [1,0]

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)
   
    anisotropicMaps = []
    anisotropicMaps.push(tf.onesLike(tempMap)) // [1,2,0], [   ]
    anisotropicMaps.push(tf.onesLike(tempMap)) // [1,2,0], [  2]
    anisotropicMaps.push(tf.onesLike(tempMap)) // [1,2,0], [  0]
    anisotropicMaps.push(tf.onesLike(tempMap)) // [1,2,0], [2,0]

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)

    anisotropicMaps = []
    anisotropicMaps.push(tf.clone(tempMap)) // [0,1,2], [   ]
    anisotropicMaps.push(tf.onesLike(tempMap)) // [0,1,2], [  1]
    anisotropicMaps.push(tf.onesLike(tempMap)) // [0,1,2], [  2]
    anisotropicMaps.push(tf.onesLike(tempMap)) // [0,1,2], [1,2]

    extendedMaps.push(anisotropicBidirectionalShadowMap(anisotropicMaps as Array4<tf.Tensor3D>))
    tf.dispose(anisotropicMaps)

    const shadowMap = extendedAnisotropicBidirectionalShadowMap(extendedMaps as Array3<tf.Tensor3D>)
    if (verbose) logExtendedAnisotropicBidirectionalShadowMaps(shadowMap)
        
    tf.dispose(extendedMaps)
    tf.dispose(tempMap)

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
    uniforms?: number[][], 
    preventEagerUnpackingOfOutput?: boolean): tf.Tensor
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(prog, inputs, dtype, uniforms, preventEagerUnpackingOfOutput)
    return tf.engine().makeTensorFromTensorInfo(info) 
}

