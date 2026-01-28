import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

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
        this.outputShape = [4, outDepth, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float avg3(float a, float b, float c) { return (a + b + c) * (1.0/3.0); }
        float max3(float a, float b, float c) { return max(max(a, b), c); }
        float min4(float a, float b, float c, float d) { return min(min(min(a, b), c), d); }

        ivec3 flip(ivec3 v, int i)
        {
            v.x = (i / 2 == 0) ? v.x : 1 - v.x;
            v.y = (i % 2 == 0) ? v.y : 1 - v.y;
            return v;
        }

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

        ivec4 getCoords()
        {
            ivec6 coords = getOutputCoords();
            return ivec4(coords.w, coords.z, coords.y, coords.x);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec4 coords)
        {
            CellValues c;

            c.v000 = getA(coords.xyz - flip(ivec3(1,1,1), coords.w));
            c.v100 = getA(coords.xyz - flip(ivec3(0,1,1), coords.w));
            c.v010 = getA(coords.xyz - flip(ivec3(1,0,1), coords.w));
            c.v001 = getA(coords.xyz - flip(ivec3(1,1,0), coords.w));
            c.v011 = getA(coords.xyz - flip(ivec3(1,0,0), coords.w));
            c.v101 = getA(coords.xyz - flip(ivec3(0,1,0), coords.w));
            c.v110 = getA(coords.xyz - flip(ivec3(0,0,1), coords.w));
            c.v111 = getA(coords.xyz - flip(ivec3(0,0,0), coords.w));

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
            ivec4 coord = getCoords();
            CellValues c = getValues(coord);

            float xMin = getMinOnFaceX(c);
            float yMin = getMinOnFaceY(c);
            float zMin = getMinOnFaceZ(c);

            setOutput(vec4(xMin, yMin, zMin, 0.0));
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
        this.outputShape = [4, outDepth, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float avg3(float a, float b, float c) { return (a + b + c) * (1.0 / 3.0); }

        ivec3 flip(ivec3 v, int i)
        {
            v.x = (i / 2 == 0) ? v.x : 1 - v.x;
            v.y = (i % 2 == 0) ? v.y : 1 - v.y;
            return v;
        }

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

        ivec4 getCoords()
        {
            ivec6 coords = getOutputCoords();
            return ivec4(coords.w, coords.z, coords.y, coords.x);
        }

        float getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x);
        }

        CellValues getValues(ivec4 coords)
        {
            CellValues c;

            c.v000 = getA(coords.xyz - flip(ivec3(1,1,1), coords.w));
            c.v100 = getA(coords.xyz - flip(ivec3(0,1,1), coords.w));
            c.v010 = getA(coords.xyz - flip(ivec3(1,0,1), coords.w));
            c.v001 = getA(coords.xyz - flip(ivec3(1,1,0), coords.w));
            c.v011 = getA(coords.xyz - flip(ivec3(1,0,0), coords.w));
            c.v101 = getA(coords.xyz - flip(ivec3(0,1,0), coords.w));
            c.v110 = getA(coords.xyz - flip(ivec3(0,0,1), coords.w));
            c.v111 = getA(coords.xyz - flip(ivec3(0,0,0), coords.w));

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
            ivec4 coords = getCoords();
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
        this.outputShape = [4, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(3, ${outWidth-1}, ${outHeight-1});

        float min3(float a, float b, float c) { return min(min(a, b), c); }

        ivec3 flip(ivec3 v, int i)
        {
            v.x = (i / 2 == 0) ? v.x : 1 - v.x;
            v.y = (i % 2 == 0) ? v.y : 1 - v.y;
            return v;
        }

        ivec3 getCoords()
        {
            ivec5 coords = getOutputCoords();
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
            ivec3 coords = getCoords();

            vec4 c111 = getA(coords - flip(ivec3(0,0,0), coords.z));
            vec4 c011 = getA(coords - flip(ivec3(1,0,0), coords.z));
            vec4 c101 = getA(coords - flip(ivec3(0,1,0), coords.z));
            vec4 c001 = getA(coords - flip(ivec3(1,1,0), coords.z));

            vec4 c110 = getB(coords - flip(ivec3(0,0,0), coords.z));
            vec4 c010 = getB(coords - flip(ivec3(1,0,0), coords.z));
            vec4 c100 = getB(coords - flip(ivec3(0,1,0), coords.z));
            vec4 c000 = getB(coords - flip(ivec3(1,1,0), coords.z));

            c111.x = getMinOnFaceX(c111, c110, c101, c100);
            c111.y = getMinOnFaceY(c111, c110, c011, c010);
            c111.z = getMinOnFaceZ(c111, c110, c101, c011, c100, c010, c001, c000);

            setOutput(c111);
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

        const ivec4 minCoords = ivec4(0);
        const ivec4 maxCoords = ivec4(3, ${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        float max3(float a, float b, float c) { return max(max(a, b), c); }

        ivec3 getCoords()
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        vec4 getA(ivec4 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.w, coords.z, coords.y, coords.x, 0, 0);
        }

        vec4 getB(ivec4 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getB(coords.w, coords.z, coords.y, coords.x, 0, 0);
        }

        int getOcclusion(vec4 minValues, vec4 maxValues)
        {
            bvec4 occ = greaterThanEqual(minValues, maxValues);
            return int(all(occ.xyz) || occ.w);
        }
                
        void main()
        {
            ivec3 coords = getCoords();
            int bitpack = 0;

            for (int i = 0; i < 4; i++)
            {
                vec4 minValues = getA(ivec4(coords, i));
                vec4 maxValues = getB(ivec4(coords, i));
                int occlusion = getOcclusion(minValues, maxValues);
                
                bitpack += occlusion << i;
            }

            setOutput(float(bitpack));
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

export function computeOcclusionMap_(volumeMap: tf.Tensor3D) : tf.Tensor<tf.Rank>
{
    const minimaProgram = new GPGPUMinimaMap(volumeMap.shape)
    const maximaProgram = new GPGPUMaximaMap(volumeMap.shape)
    const updateProgram = new GPGPUUpdateSlice(volumeMap.shape)
    const occlusionProgram = new GPGPUOcclusionMap(volumeMap.shape)

    let minima = runProgram(minimaProgram, [volumeMap])
    let slices = tf.unstack(minima, 1)
    minima.dispose()

    for (let i = 1; i < slices.length; i++)
    {
        const updatedSlice = runProgram(updateProgram, [slices[i], slices[i-1]])
        tf.dispose(slices[i])
        slices[i] = updatedSlice
    }

    minima = tf.stack(slices, 1); 
    tf.dispose(slices)

    const maxima = runProgram(maximaProgram, [volumeMap])
    const occlusion = runProgram(occlusionProgram, [minima, maxima])
    tf.dispose([minima, maxima])

    console.log('occlusionMap', tf.tidy(() => occlusion.floorDiv(1).mod(2).mean().dataSync()))
    console.log('occlusionMap', tf.tidy(() => occlusion.floorDiv(2).mod(2).mean().dataSync()))
    console.log('occlusionMap', tf.tidy(() => occlusion.floorDiv(4).mod(2).mean().dataSync()))
    console.log('occlusionMap', tf.tidy(() => occlusion.floorDiv(8).mod(2).mean().dataSync()))
    // console.log('occlusionMap', tf.tidy(() => occlusion.mean().dataSync()))

    return occlusion as tf.Tensor
}

type Axis = 0 | 1 | 2;
type Reverses =  Axis[];
type Permutations = [Axis, Axis, Axis];

export function computeOcclusionMap(volumeMap: tf.Tensor3D): tf.Tensor3D
{
    const occlusion0 = tf.tidy(() => computeOcclusionMap_(volumeMap))
    const occlusion1 = tf.tidy(() => tf.reverse(computeOcclusionMap_(tf.reverse(volumeMap))))

    const occlusionMap = tf.maximum(occlusion0, occlusion1)
    tf.dispose([occlusion0, occlusion1])

    console.log('occlusionMap', occlusionMap.mean().dataSync())

    return occlusionMap as tf.Tensor3D
}
