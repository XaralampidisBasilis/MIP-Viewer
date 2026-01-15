import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

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

            vec4 f000 = getPackedValues(voxelCoords - ivec3(0,0,0));
            vec4 f001 = getPackedValues(voxelCoords - ivec3(0,0,1));
            vec4 v201 = getPackedValues(voxelCoords - ivec3(2,0,1));
            vec4 v021 = getPackedValues(voxelCoords - ivec3(0,2,1));
            vec4 v221 = getPackedValues(voxelCoords - ivec3(2,2,1));

            f000.r = max(f000.r, mmin(f001.r, v201.g, v021.b, v221.a));
            f000.g = max(f000.g, mmin(f001.r, f001.g, v021.b, v021.a));
            f000.b = max(f000.b, mmin(f001.r, v201.g, f001.b, v201.a));
            f000.a = max(f000.a, mmin(f001.r, f001.g, f001.b, f001.a));

            setOutput(f000);
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

            vec4 f000 = getPackedValues(voxelCoords + ivec3(0,0,0));
            vec4 f001 = getPackedValues(voxelCoords + ivec3(0,0,1));
            vec4 v201 = getPackedValues(voxelCoords + ivec3(2,0,1));
            vec4 v021 = getPackedValues(voxelCoords + ivec3(0,2,1));
            vec4 v221 = getPackedValues(voxelCoords + ivec3(2,2,1));

            f000.r = max(f000.r, mmin(f001.r, f001.g, f001.b, f001.a));
            f000.g = max(f000.g, mmin(v201.r, f001.g, v201.b, f001.a));
            f000.b = max(f000.b, mmin(v021.r, v021.g, f001.b, f001.a));
            f000.a = max(f000.a, mmin(v221.r, v021.g, v201.b, f001.a));

            setOutput(f000);
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
            
            float f000 = getVoxelValue(voxelCoords + ivec3(0,0,0));
            float f100 = getVoxelValue(voxelCoords + ivec3(1,0,0));
            float f010 = getVoxelValue(voxelCoords + ivec3(0,1,0));
            float f001 = getVoxelValue(voxelCoords + ivec3(0,0,1));
            float f011 = getVoxelValue(voxelCoords + ivec3(0,1,1));
            float f101 = getVoxelValue(voxelCoords + ivec3(1,0,1));
            float f110 = getVoxelValue(voxelCoords + ivec3(1,1,0));
            float f111 = getVoxelValue(voxelCoords + ivec3(1,1,1));

            float m000 = getPropagatedValue(voxelCoords + ivec3(0,0,0));
            float m010 = getPropagatedValue(voxelCoords + ivec3(0,1,0));
            float m100 = getPropagatedValue(voxelCoords + ivec3(1,0,0));
            float m110 = getPropagatedValue(voxelCoords + ivec3(1,1,0));
            float m001 = getPropagatedValue(voxelCoords + ivec3(0,0,1));
            float m011 = getPropagatedValue(voxelCoords + ivec3(0,1,1));
            float m101 = getPropagatedValue(voxelCoords + ivec3(1,0,1));
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
            
            float f000 = getVoxelValue(voxelCoords + ivec3(0,0,0));
            float f100 = getVoxelValue(voxelCoords + ivec3(1,0,0));
            float f010 = getVoxelValue(voxelCoords + ivec3(0,1,0));
            float f001 = getVoxelValue(voxelCoords + ivec3(0,0,1));
            float f011 = getVoxelValue(voxelCoords + ivec3(0,1,1));
            float f101 = getVoxelValue(voxelCoords + ivec3(1,0,1));
            float f110 = getVoxelValue(voxelCoords + ivec3(1,1,0));
            float f111 = getVoxelValue(voxelCoords + ivec3(1,1,1));

            float m111 = getPropagatedValue(voxelCoords + ivec3(1,1,1));
            float m101 = getPropagatedValue(voxelCoords + ivec3(1,0,1));
            float m011 = getPropagatedValue(voxelCoords + ivec3(0,1,1));
            float m001 = getPropagatedValue(voxelCoords + ivec3(0,0,1));
            float m110 = getPropagatedValue(voxelCoords + ivec3(1,1,0));
            float m100 = getPropagatedValue(voxelCoords + ivec3(1,0,0));
            float m010 = getPropagatedValue(voxelCoords + ivec3(0,1,0));
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

        if (i%2==0) await tf.nextFrame()                     
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
