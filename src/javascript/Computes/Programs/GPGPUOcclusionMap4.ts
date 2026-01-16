import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class GPGPUDualPropagationMap implements GPGPUProgram 
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

            bool insideWidth  = safeCoords.x < ${inWidth-1};
            bool insideHeight = safeCoords.y < ${inHeight-1};

            vec4 packedValues = getA(safeCoords.z, safeCoords.y, safeCoords.x);
            packedValues.ga = insideWidth ? packedValues.ga : packedValues.rb;
            packedValues.ba = insideHeight ? packedValues.ba : packedValues.rg;

            return packedValues;
        }

        void main()
        {
            ivec3 voxelCoords = getVoxelCoords();

            vec4 v000 = getPackedValues(voxelCoords);
            
            vec4 n001 = getPackedValues(voxelCoords - ivec3(0,0,1));
            vec4 n201 = getPackedValues(voxelCoords - ivec3(2,0,1));
            vec4 n021 = getPackedValues(voxelCoords - ivec3(0,2,1));
            vec4 n221 = getPackedValues(voxelCoords - ivec3(2,2,1));
            vec4 p001 = getPackedValues(voxelCoords + ivec3(0,0,1));
            vec4 p201 = getPackedValues(voxelCoords + ivec3(2,0,1));
            vec4 p021 = getPackedValues(voxelCoords + ivec3(0,2,1));
            vec4 p221 = getPackedValues(voxelCoords + ivec3(2,2,1));

            v000.r = max(v000.r, mmin(n001.r, n201.g, n021.b, n221.a));
            v000.g = max(v000.g, mmin(n001.r, n001.g, n021.b, n021.a));
            v000.b = max(v000.b, mmin(n001.r, n201.g, n001.b, n201.a));
            v000.a = max(v000.a, mmin(n001.r, n001.g, n001.b, n001.a));
            v000.r = max(v000.r, mmin(p001.r, p001.g, p001.b, p001.a));
            v000.g = max(v000.g, mmin(p201.r, p001.g, p201.b, p001.a));
            v000.b = max(v000.b, mmin(p021.r, p021.g, p001.b, p001.a));
            v000.a = max(v000.a, mmin(p221.r, p021.g, p201.b, p001.a));

            setOutput(v000);
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

            bool insideWidth  = safeCoords.x < ${inWidth-1};
            bool insideHeight = safeCoords.y < ${inHeight-1};

            vec4 packedValues = getA(safeCoords.z, safeCoords.y, safeCoords.x);
            packedValues.ga = insideWidth ? packedValues.ga : packedValues.rb;
            packedValues.ba = insideHeight ? packedValues.ba : packedValues.rg;

            return packedValues;
        }

        void main()
        {
            ivec3 voxelCoords = getVoxelCoords();

            vec4 v000 = getPackedValues(voxelCoords - ivec3(0,0,0));
            vec4 v001 = getPackedValues(voxelCoords - ivec3(0,0,1));
            vec4 v201 = getPackedValues(voxelCoords - ivec3(2,0,1));
            vec4 v021 = getPackedValues(voxelCoords - ivec3(0,2,1));
            vec4 v221 = getPackedValues(voxelCoords - ivec3(2,2,1));

            v000.r = max(v000.r, mmin(v001.r, v201.g, v021.b, v221.a));
            v000.g = max(v000.g, mmin(v001.r, v001.g, v021.b, v021.a));
            v000.b = max(v000.b, mmin(v001.r, v201.g, v001.b, v201.a));
            v000.a = max(v000.a, mmin(v001.r, v001.g, v001.b, v001.a));

            setOutput(v000);
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

            bool insideWidth  = safeCoords.x < ${inWidth-1};
            bool insideHeight = safeCoords.y < ${inHeight-1};

            vec4 packedValues = getA(safeCoords.z, safeCoords.y, safeCoords.x);
            packedValues.ga = insideWidth ? packedValues.ga : packedValues.rb;
            packedValues.ba = insideHeight ? packedValues.ba : packedValues.rg;

            return packedValues;
        }

        void main()
        {
            ivec3 voxelCoords = getVoxelCoords();

            vec4 v000 = getPackedValues(voxelCoords + ivec3(0,0,0));
            vec4 v001 = getPackedValues(voxelCoords + ivec3(0,0,1));
            vec4 v201 = getPackedValues(voxelCoords + ivec3(2,0,1));
            vec4 v021 = getPackedValues(voxelCoords + ivec3(0,2,1));
            vec4 v221 = getPackedValues(voxelCoords + ivec3(2,2,1));

            v000.r = max(v000.r, mmin(v001.r, v001.g, v001.b, v001.a));
            v000.g = max(v000.g, mmin(v201.r, v001.g, v201.b, v001.a));
            v000.b = max(v000.b, mmin(v021.r, v021.g, v001.b, v001.a));
            v000.a = max(v000.a, mmin(v221.r, v021.g, v201.b, v001.a));

            setOutput(v000);
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
            
            float v000 = getVoxelValue(voxelCoords + ivec3(0,0,0));
            float v100 = getVoxelValue(voxelCoords + ivec3(1,0,0));
            float v010 = getVoxelValue(voxelCoords + ivec3(0,1,0));
            float v001 = getVoxelValue(voxelCoords + ivec3(0,0,1));
            float v011 = getVoxelValue(voxelCoords + ivec3(0,1,1));
            float v101 = getVoxelValue(voxelCoords + ivec3(1,0,1));
            float v110 = getVoxelValue(voxelCoords + ivec3(1,1,0));
            float v111 = getVoxelValue(voxelCoords + ivec3(1,1,1));

            float m000 = getPropagatedValue(voxelCoords + ivec3(0,0,0));
            float m010 = getPropagatedValue(voxelCoords + ivec3(0,1,0));
            float m100 = getPropagatedValue(voxelCoords + ivec3(1,0,0));
            float m110 = getPropagatedValue(voxelCoords + ivec3(1,1,0));

            float M000 = 0.0;
            float M010 = 0.0;
            float M100 = 0.0;
            float M110 = 0.0;

            M000 = max(M000, v001);
            M000 = max(M000, v011);
            M000 = max(M000, v101);
            M000 = max(M000, v111);
            M000 = max(M000, (v000 + v100 + v001) / 3.0);
            M000 = max(M000, (v000 + v010 + v001) / 3.0);
            M000 = max(M000, (v100 + v010 + v001) / 3.0);
            M000 = max(M000, (v100 + v001 + v101) / 3.0);
            M000 = max(M000, (v010 + v001 + v011) / 3.0);
            M000 = max(M000, (v110 + v101 + v011) / 3.0);

            M010 = max(M010, v011);
            M010 = max(M010, v111);
            M010 = max(M010, (v110 + v011 + v010) / 3.0);
            M010 = max(M010, (v110 + v011 + v111) / 3.0);

            M100 = max(M100, v101);
            M100 = max(M100, v111);
            M100 = max(M100, (v110 + v101 + v100) / 3.0);
            M100 = max(M100, (v110 + v101 + v111) / 3.0);

            M110 = max(M110, v111);

            bool occlusion =
                m000 >= M000 &&
                m010 >= M010 &&
                m100 >= M100 &&
                m110 >= M110;

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
            
            float v000 = getVoxelValue(voxelCoords + ivec3(0,0,0));
            float v100 = getVoxelValue(voxelCoords + ivec3(1,0,0));
            float v010 = getVoxelValue(voxelCoords + ivec3(0,1,0));
            float v001 = getVoxelValue(voxelCoords + ivec3(0,0,1));
            float v011 = getVoxelValue(voxelCoords + ivec3(0,1,1));
            float v101 = getVoxelValue(voxelCoords + ivec3(1,0,1));
            float v110 = getVoxelValue(voxelCoords + ivec3(1,1,0));
            float v111 = getVoxelValue(voxelCoords + ivec3(1,1,1));

            float m111 = getPropagatedValue(voxelCoords + ivec3(1,1,1));
            float m101 = getPropagatedValue(voxelCoords + ivec3(1,0,1));
            float m011 = getPropagatedValue(voxelCoords + ivec3(0,1,1));
            float m001 = getPropagatedValue(voxelCoords + ivec3(0,0,1));

            float M111 = 0.0;
            float M101 = 0.0;
            float M011 = 0.0;
            float M001 = 0.0;

            M111 = max(M111, v110);
            M111 = max(M111, v100);
            M111 = max(M111, v010);
            M111 = max(M111, v000);
            M111 = max(M111, (v111 + v011 + v110) / 3.0);
            M111 = max(M111, (v111 + v101 + v110) / 3.0);
            M111 = max(M111, (v011 + v101 + v110) / 3.0);
            M111 = max(M111, (v011 + v110 + v010) / 3.0);
            M111 = max(M111, (v101 + v110 + v100) / 3.0);
            M111 = max(M111, (v001 + v010 + v100) / 3.0);

            M101 = max(M101, v100);
            M101 = max(M101, v000);
            M101 = max(M101, (v001 + v100 + v101) / 3.0);
            M101 = max(M101, (v001 + v100 + v000) / 3.0);

            M011 = max(M011, v010);
            M011 = max(M011, v000);
            M011 = max(M011, (v001 + v010 + v011) / 3.0);
            M011 = max(M011, (v001 + v010 + v000) / 3.0);

            M001 = max(M001, v000);

            bool occlusion =
                m111 >= M111 &&
                m101 >= M101 &&
                m011 >= M011 &&
                m001 >= M001;

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
    const occlusionProgram = new GPGPUBackOcclusionMap(volumeMap.shape)

    let propagationMap = runProgram(propagationProgram, [volumeMap])
    let maxPropagation = Math.max(...volumeMap.shape) / 2

    for (let i = 0; i < maxPropagation; i++) 
    {
        const prev = propagationMap
        propagationMap = runProgram(propagationProgram, [prev])
        prev.dispose()

        if (i%10==0) await tf.nextFrame()                     
    }

    const occlusionMap = runProgram(occlusionProgram, [volumeMap, propagationMap], 'bool')
    propagationMap.dispose()

    return occlusionMap as tf.Tensor
}

async function computeFrontOcclusionMap(volumeMap: tf.Tensor3D) : Promise<tf.Tensor<tf.Rank>>
{
    const propagationProgram = new GPGPUFrontPropagationMap(volumeMap.shape)
    const occlusionProgram = new GPGPUFrontOcclusionMap(volumeMap.shape)

    let propagationMap = runProgram(propagationProgram, [volumeMap])
    let maxPropagation = Math.max(...volumeMap.shape) / 2

    for (let i = 0; i < maxPropagation; i++) 
    {
        const prev = propagationMap
        propagationMap = runProgram(propagationProgram, [prev])
        prev.dispose()

        if (i%10==0) await tf.nextFrame()                     
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

export async function computeOcclusionMap2(volumeMap: tf.Tensor3D) : Promise<tf.Tensor<tf.Rank>>
{
    const dualPropagationProgram = new GPGPUDualPropagationMap(volumeMap.shape)
    const backOcclusionProgram = new GPGPUBackOcclusionMap(volumeMap.shape)
    const frontOcclusionProgram = new GPGPUFrontOcclusionMap(volumeMap.shape)

    let dualPropagationMap = runProgram(dualPropagationProgram, [volumeMap])
    let maxPropagation = Math.max(...volumeMap.shape) / 2

    for (let i = 0; i < maxPropagation; i++) 
    {
        const prev = dualPropagationMap
        dualPropagationMap = runProgram(dualPropagationProgram, [prev])
        prev.dispose()

        if (i%10==0) await tf.nextFrame()                     
    }

    const backOcclusionMap = runProgram(backOcclusionProgram, [volumeMap, dualPropagationMap], 'bool')
    const frontOcclusionMap = runProgram(frontOcclusionProgram, [volumeMap, dualPropagationMap], 'bool')
    dualPropagationMap.dispose()

    const occlusionMap = tf.logicalOr(backOcclusionMap, frontOcclusionMap);
    tf.dispose([backOcclusionMap, frontOcclusionMap]);

    return occlusionMap as tf.Tensor
}
