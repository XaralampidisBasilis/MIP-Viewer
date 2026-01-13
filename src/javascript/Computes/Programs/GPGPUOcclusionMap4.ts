import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class GPGPUStartMap implements GPGPUProgram 
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
        this.outputShape = [inDepth, inHeight, inWidth, 2, 2]  
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        ivec3 getVoxelCoords()
        {
            ivec5 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        float getVoxelValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x);
        }

        void main()
        {
            ivec3 voxelCoords = getVoxelCoords();
            setOutput(vec4(getVoxelValue(voxelCoords)));
        }
        `
    }
}

class GPGPUPropagationMap implements GPGPUProgram 
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
        this.outputShape = [inDepth, inHeight, inWidth, 2, 2]
        this.userCode = `

        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        float mmin(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        ivec3 getVoxelCoords()
        {
            ivec5 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        vec4 getVoxelValue(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x, 0, 0);
        }

        void main()
        {
            ivec3 voxelCoords = getVoxelCoords();

            vec4 C000 = getVoxelValue(voxelCoords);

            float G100 = getVoxelValue(voxelCoords - ivec3(1,0,0)).g;
            float G110 = getVoxelValue(voxelCoords - ivec3(1,1,0)).g;
            float G101 = getVoxelValue(voxelCoords - ivec3(1,0,1)).g;
            float G111 = getVoxelValue(voxelCoords - ivec3(1,1,1)).g;

            float B100 = getVoxelValue(voxelCoords + ivec3(1,0,0)).b;
            float B110 = getVoxelValue(voxelCoords + ivec3(1,1,0)).b;
            float B101 = getVoxelValue(voxelCoords + ivec3(1,0,1)).b;
            float B111 = getVoxelValue(voxelCoords + ivec3(1,1,1)).b;

            C000.g = max(C000.g, mmin(G100, G110, G101, G111));
            C000.b = max(C000.b, mmin(B100, B110, B101, B111));

            setOutput(vec4(C000));
        }
        `
    }
}

class GPGPUOcclusionMap implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = true
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

        float mmin(float a, float b, float c, float d)
        {
            return min(min(min(a, b), c), d);
        }

        float mmax(float a, float b, float c, float d)
        {
            return max(max(max(a, b), c), d);
        }

        ivec3 getCellCoords()
        {
            ivec3 outCoords = getOutputCoords();
            return ivec3(outCoords.z, outCoords.y, outCoords.x);
        }

        vec4 getVoxelValues(ivec3 voxelCoords)
        {
            ivec3 safeCoords = clamp(voxelCoords, minCoords, maxCoords);
            return getA(safeCoords.z, safeCoords.y, safeCoords.x, 0, 0);
        }

        void main()
        {
            ivec3 cellCoords = getCellCoords();
            ivec3 voxelCoords = cellCoords - 1;
            
            vec4 C000 = getVoxelValues(voxelCoords + ivec3(0,0,0));
            vec4 C100 = getVoxelValues(voxelCoords + ivec3(1,0,0));
            vec4 C010 = getVoxelValues(voxelCoords + ivec3(0,1,0));
            vec4 C001 = getVoxelValues(voxelCoords + ivec3(0,0,1));
            vec4 C011 = getVoxelValues(voxelCoords + ivec3(0,1,1));
            vec4 C101 = getVoxelValues(voxelCoords + ivec3(1,0,1));
            vec4 C110 = getVoxelValues(voxelCoords + ivec3(1,1,0));
            vec4 C111 = getVoxelValues(voxelCoords + ivec3(1,1,1));


            // POSITIVE DIRECTION
            //________________________________________________________

            float U000 = 0.0;
            float U010 = 0.0;
            float U001 = 0.0;
            float U011 = 0.0;
   
            U000 = max(U000, C100.r);
            U000 = max(U000, C110.r);
            U000 = max(U000, C101.r);
            U000 = max(U000, C111.r);
            U000 = max(U000, (C000.r + C001.r + C100.r) / 3.0);
            U000 = max(U000, (C000.r + C010.r + C100.r) / 3.0);
            U000 = max(U000, (C001.r + C010.r + C100.r) / 3.0);
            U000 = max(U000, (C001.r + C100.r + C101.r) / 3.0);
            U000 = max(U000, (C010.r + C100.r + C110.r) / 3.0);
            U000 = max(U000, (C011.r + C101.r + C110.r) / 3.0);

            U010 = max(U010, C110.r);
            U010 = max(U010, C111.r);
            U010 = max(U010, (C011.r + C110.r + C010.r) / 3.0);
            U010 = max(U010, (C011.r + C110.r + C111.r) / 3.0);

            U001 = max(U001, C101.r);
            U001 = max(U001, C111.r);
            U001 = max(U001, (C011.r + C101.r + C001.r) / 3.0);
            U001 = max(U001, (C011.r + C101.r + C111.r) / 3.0);

            U011 = max(U011, C111.r);

            bool O000 = C000.g >= U000; 
            bool O010 = C010.g >= U010; 
            bool O001 = C001.g >= U001; 
            bool O011 = C011.g >= U011;

            bool O0 = O000 && O010 && O001 && O011;


            // NEGATIVE DIRECTION
            //________________________________________________________

            float U111 = 0.0;
            float U101 = 0.0;
            float U110 = 0.0;
            float U100 = 0.0;

            U111 = max(U111, C011.r);
            U111 = max(U111, C001.r);
            U111 = max(U111, C010.r);
            U111 = max(U111, C000.r);
            U111 = max(U111, (C111.r + C110.r + C011.r) / 3.0);
            U111 = max(U111, (C111.r + C101.r + C011.r) / 3.0);
            U111 = max(U111, (C110.r + C101.r + C011.r) / 3.0);
            U111 = max(U111, (C110.r + C011.r + C010.r) / 3.0);
            U111 = max(U111, (C101.r + C011.r + C001.r) / 3.0);
            U111 = max(U111, (C100.r + C010.r + C001.r) / 3.0);

            U101 = max(U101, C001.r);
            U101 = max(U101, C000.r);
            U101 = max(U101, (C100.r + C001.r + C101.r) / 3.0);
            U101 = max(U101, (C100.r + C001.r + C000.r) / 3.0);

            U110 = max(U110, C010.r);
            U110 = max(U110, C000.r);
            U110 = max(U110, (C100.r + C010.r + C110.r) / 3.0);
            U110 = max(U110, (C100.r + C010.r + C000.r) / 3.0);

            U100 = max(U100, C000.r);

            bool O111 = C111.b >= U111; 
            bool O101 = C101.b >= U101; 
            bool O110 = C110.b >= U110; 
            bool O100 = C100.b >= U100;

            bool O1 = O111 && O101 && O110 && O100;

            
            // COMBINED OCCLUSIONS

            setOutput(float(O0 || O1));
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

export async function computeOcclusionMap(volumeMap: tf.Tensor3D) : Promise<tf.Tensor<tf.Rank>>
{
    const startProgram = new GPGPUStartMap(volumeMap.shape)
    const propagationProgram = new GPGPUPropagationMap(volumeMap.shape)
    const occlusionProgram = new GPGPUOcclusionMap(volumeMap.shape)

    let propagationMap = runProgram(startProgram, [volumeMap])
    let maxPropagation = Math.ceil(Math.max(...volumeMap.shape) * 0.4)

    for (let i = 0; i < maxPropagation; i++) 
    {
        console.log(i)
        const prev = propagationMap
        propagationMap = runProgram(propagationProgram, [prev])
        prev.dispose()

        await tf.nextFrame()                     
    }

    const occlusionMap = runProgram(occlusionProgram, [propagationMap])
    propagationMap.dispose()

    return occlusionMap as tf.Tensor
}