import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class GPGPUMinDiffMap implements GPGPUProgram 
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

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        ivec3 getInCoords(ivec3 coords, int ox, int oy, int oz)
        {
            return coords + ivec3(ox, oy, oz);
        }

        ivec3 getCellCoords(ivec3 coords, int ox, int oy, int oz)
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

        float getMinDiffOnFaceX(CellValues c)
        {
            float m = 1.0;

            m = min(m, c.v000 - c.v111);
            m = min(m, c.v010 - c.v111);
            m = min(m, c.v100 - c.v111);
            m = min(m, c.v110 - c.v111);
            m = min(m, c.v000 - c.v101);
            m = min(m, c.v100 - c.v101);

            return max(m, 0.0);
        }

        float getMinDiffOnFaceY(CellValues c)
        {
            float m = 1.0;

            m = min(m, c.v000 - c.v111);
            m = min(m, c.v100 - c.v111);
            m = min(m, c.v010 - c.v111);
            m = min(m, c.v110 - c.v111);
            m = min(m, c.v000 - c.v011);
            m = min(m, c.v010 - c.v011);

            return max(m, 0.0);
        }
            
        float getMinDiffOnFaceZ(CellValues c)
        {
            float m = 1.0;

            m = min(m, c.v000 - c.v111);
            m = min(m, c.v010 - c.v111);
            m = min(m, c.v100 - c.v111);
            m = min(m, c.v110 - c.v111);
            m = min(m, c.v000 - c.v101);
            m = min(m, c.v100 - c.v101);
            m = min(m, c.v000 - c.v011);
            m = min(m, c.v010 - c.v011);
            m = min(m, c.v000 - c.v001);

            return max(m, 0.0);
        }

        void main()
        {
            ivec3 coords = getOutCoords();
            CellValues c = getValues(coords);
        
            float xMin = getMinDiffOnFaceX(c);
            float yMin = getMinDiffOnFaceY(c);
            float zMin = getMinDiffOnFaceZ(c);

            setOutput(vec4(xMin, yMin, zMin, 0.0));
        }
        `
    }
}

class GPGPUMaxDiffMap implements GPGPUProgram 
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

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        ivec3 getInCoords(ivec3 coords, int ox, int oy, int oz)
        {
            return  coords + ivec3(ox, oy, oz);
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
    
        float getMaxOnFaceY(CellValues c)
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

        float getMaxOnFaceZ(CellValues c)
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
            ivec3 cellCoords = getOutCoords();
            CellValues c = getValues(cellCoords);

            float xMax = getMaxOnFaceX(c);
            float yMax = getMaxOnFaceY(c);
            float zMax = getMaxOnFaceZ(c);

            setOutput(vec4(xMax, yMax, zMax, 0.0));
        }
        `
    }
}

class GPGPUUpdateMinDiffSlices implements GPGPUProgram 
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

        ivec2 getOutCoords()
        {
            ivec4 coords = getOutputCoords();
            return ivec2(coords.y, coords.x);
        }

        ivec2 getInCoords(ivec2 coords, int ox, int oy)
        {
            return coords + ivec2(ox, oy);
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
            
            t10 = max(c101.y + t10, 0.0);
            t10 = min(c110.z,  t10);
            t10 = max(c111.x + t10, 0.0);

            return t10;
        }

        float getMinOnFaceY(vec4 c111, vec4 c110, vec4 c011, vec4 c010)
        {
            float t01 = c010.z;

            t01 = max(c011.x + t01, 0.0);
            t01 = min(c110.z,  t01);
            t01 = max(c111.y + t01, 0.0);

            return t01;
        }

        float getMinOnFaceZ(vec4 c111, vec4 c110, vec4 c101, vec4 c011, vec4 c100, vec4 c010, vec4 c001, vec4 c000)
        {
            float t00, t01, t10, t11;
            t00 = c000.z;

            t01 = max(c001.y + t00, 0.0);
            t01 = min(c010.z,  t01);
            t01 = max(c011.x + t01, 0.0);

            t10 = max(c001.x + t00, 0.0);
            t10 = min(c100.z,  t10);
            t10 = max(c101.y + t10, 0.0);

            t11 = min(t01, t10);
            t11 = min(c110.z,  t11);
            t11 = max(c111.z + t11, 0.0);

            return t11;
        }
                
        void main()
        {
            ivec2 cellCoords = getOutCoords();

            vec4 c111 = getA(getInCoords(cellCoords, -0,-0));
            vec4 c011 = getA(getInCoords(cellCoords, -1,-0));
            vec4 c101 = getA(getInCoords(cellCoords, -0,-1));
            vec4 c001 = getA(getInCoords(cellCoords, -1,-1));
            vec4 c110 = getB(getInCoords(cellCoords, -0,-0));
            vec4 c010 = getB(getInCoords(cellCoords, -1,-0));
            vec4 c100 = getB(getInCoords(cellCoords, -0,-1));
            vec4 c000 = getB(getInCoords(cellCoords, -1,-1));

            c111.x = getMinOnFaceX(c111, c110, c101, c100);
            c111.y = getMinOnFaceY(c111, c110, c011, c010);
            c111.z = getMinOnFaceZ(c111, c110, c101, c011, c100, c010, c001, c000);

            setOutput(c111);
        }
        `
    }
}

class GPGPUUpdateMinDiffMap implements GPGPUProgram 
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
        this.outputShape = [outDepth, outHeight, outWidth, 2, 2]     
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${outWidth-1}, ${outHeight-1}, ${outDepth-1});

        float min3(float a, float b, float c) { return min(min(a, b), c); }

        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        ivec3 getInCoords(ivec3 coords, int ox, int oy, int oz)
        {
            return coords + ivec3(ox, oy, oz);
        }

        vec4 getA(ivec3 coords)
        {
            coords = clamp(coords, minCoords, maxCoords);
            return getA(coords.z, coords.y, coords.x, 0, 0);
        }


        float getMinOnFaceX(vec4 c111, vec4 c110, vec4 c101, vec4 c100)
        {
            float t10 = c100.z;
            
            t10 = max(c101.y + t10, 0.0);
            t10 = min(c110.z, t10);
            t10 = max(c111.x + t10, 0.0);

            return t10;
        }

        float getMinOnFaceY(vec4 c111, vec4 c110, vec4 c011, vec4 c010)
        {
            float t01 = c010.z;

            t01 = max(c011.x + t01, 0.0);
            t01 = min(c110.z, t01);
            t01 = max(c111.y + t01, 0.0);

            return t01;
        }

        float getMinOnFaceZ(vec4 c111, vec4 c110, vec4 c101, vec4 c011, vec4 c100, vec4 c010, vec4 c001, vec4 c000)
        {
            float t00, t01, t10, t11;

            t00 = c000.z;

            t01 = max(c001.y + t00, 0.0);
            t01 = min(c010.z, t01);
            t01 = max(c011.x + t01, 0.0);

            t10 = max(c001.x + t00, 0.0);
            t10 = min(c100.z, t10);
            t10 = max(c101.y + t10, 0.0);

            t11 = min(t01, t10);
            t11 = min(c110.z, t11);
            t11 = max(c111.z + t11, 0.0);

            return t11;
        }
                
        void main()
        {
            ivec3 coords = getOutCoords();

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
            ivec3 cellCoords = getOutCoords();

            setOutput(float(getOcclusion(cellCoords)));
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

export function computeOneWayOcclusionMap(volumeMap: tf.Tensor3D) : tf.Tensor<tf.Rank>
{
    const minimaProgram = new GPGPUMinDiffMap(volumeMap.shape)
    const minimaStart = runProgram(minimaProgram, [volumeMap])
    console.log('minimaStart', tf.tidy(() => minimaStart.mean([0,1,2]).dataSync())) 

    const updateProgram = new GPGPUUpdateMinDiffSlices(volumeMap.shape)
    const slices = tf.unstack(minimaStart, 0)
    minimaStart.dispose()

    for (let i = 1; i < slices.length; i++)
    {
        const updatedSlice = runProgram(updateProgram, [slices[i], slices[i-1]])
        tf.dispose(slices[i])
        slices[i] = updatedSlice
    }

    const minima = tf.stack(slices, 0)
    tf.dispose(slices)
    console.log('minima', tf.tidy(() => minima.mean([0,1,2]).dataSync())) 
    
    const maximaProgram = new GPGPUMaxDiffMap(volumeMap.shape)
    const maxima = runProgram(maximaProgram, [volumeMap])
    console.log('maxima', tf.tidy(() => maxima.mean([0,1,2]).dataSync())) 

    const occlusionProgram = new GPGPUOcclusionMap(volumeMap.shape)
    const occlusion = runProgram(occlusionProgram, [minima, maxima])
    tf.dispose([minima, maxima])
    console.log('occlusion', tf.tidy(() => occlusion.mean().dataSync()))

    return occlusion as tf.Tensor
}

export async function computeOneWayOcclusionMap2(volumeMap: tf.Tensor3D) : Promise<tf.Tensor<tf.Rank>>
{
    const updateProgram = new GPGPUUpdateMinDiffMap(volumeMap.shape)
    const minimaProgram = new GPGPUMinDiffMap(volumeMap.shape)
    let minima = runProgram(minimaProgram, [volumeMap])
    console.log('minima', tf.tidy(() => minima.mean([0,1,2]).dataSync())) 

    for (let i = 1; i < volumeMap.shape[0]; i++)
    {
        const temp = runProgram(updateProgram, [minima])
        tf.dispose(minima)
        minima = temp

        await tf.nextFrame()
    }
    console.log('minima', tf.tidy(() => minima.mean([0,1,2]).dataSync())) 

    const maximaProgram = new GPGPUMaxDiffMap(volumeMap.shape)
    const maxima = runProgram(maximaProgram, [volumeMap])
    console.log('maxima', tf.tidy(() => maxima.mean([0,1,2]).dataSync())) 

    const occlusionProgram = new GPGPUOcclusionMap(volumeMap.shape)
    const occlusion = runProgram(occlusionProgram, [minima, maxima])
    tf.dispose([minima, maxima])
    console.log('occlusion', tf.tidy(() => occlusion.mean().dataSync()))

    return occlusion as tf.Tensor
}

export function computeOcclusionMap(volumeMap: tf.Tensor3D): tf.Tensor3D
{
    const occlusionMapPos = computeOneWayOcclusionMap(volumeMap)
    const occlusionMapNeg = tf.tidy(() => tf.reverse(computeOneWayOcclusionMap(tf.reverse(volumeMap))))

    const occlusionMap = tf.maximum(occlusionMapPos, occlusionMapNeg)
    tf.dispose([occlusionMapPos, occlusionMapNeg])
    console.log('occlusionMap', tf.tidy(() => occlusionMap.mean().dataSync()))

    return occlusionMap as tf.Tensor3D
}
