import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'

class GPGPUOcclusionMap implements GPGPUProgram 
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
        const [outDepth, outHeight, outWidth] = [inDepth, inHeight, inWidth].map((x: number) => x + 1)
        this.outputShape = [outDepth, outHeight, outWidth]     
        this.userCode = `

        const float INFINITY = 100000000.0;
        const ivec3 voxelMinCoords = ivec3(0);
        const ivec3 voxelMaxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        struct CellSamples
        {
            float f000;
            float f100;
            float f010;
            float f001;
            float f011;
            float f101;
            float f110;
            float f111;
        };

        ivec3 getCellCoords()
        {
            ivec3 outputCoords = getOutputCoords();
            ivec3 cellCoords = outputCoords.zyx;

            return cellCoords;
        }

        float getVoxelSample(ivec3 voxelCoords)
        {
            ivec3 voxelCoordsSafe = clamp(voxelCoords, voxelMinCoords, voxelMaxCoords);
            float voxelSample = getA(voxelCoordsSafe.z, voxelCoordsSafe.y, voxelCoordsSafe.x);

            return voxelSample;
        }

        CellSamples getCellSamples(in ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;
            
            CellSamples s;
            s.f000 = getVoxelSample(voxelCoords + ivec3(0,0,0));
            s.f100 = getVoxelSample(voxelCoords + ivec3(1,0,0));
            s.f010 = getVoxelSample(voxelCoords + ivec3(0,1,0));
            s.f001 = getVoxelSample(voxelCoords + ivec3(0,0,1));
            s.f011 = getVoxelSample(voxelCoords + ivec3(0,1,1));
            s.f101 = getVoxelSample(voxelCoords + ivec3(1,0,1));
            s.f110 = getVoxelSample(voxelCoords + ivec3(1,1,0));
            s.f111 = getVoxelSample(voxelCoords + ivec3(1,1,1));
            return s;
        }

        bool getCellOcclusion(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            float f000 = getVoxelSample(voxelCoords + ivec3(0, 0, 0));
            float f100 = getVoxelSample(voxelCoords + ivec3(1, 0, 0));
            float f010 = getVoxelSample(voxelCoords + ivec3(0, 1, 0));
            float f001 = getVoxelSample(voxelCoords + ivec3(0, 0, 1));
            float f011 = getVoxelSample(voxelCoords + ivec3(0, 1, 1));
            float f101 = getVoxelSample(voxelCoords + ivec3(1, 0, 1));
            float f110 = getVoxelSample(voxelCoords + ivec3(1, 1, 0));
            float f111 = getVoxelSample(voxelCoords + ivec3(1, 1, 1));

            // Solve for fx * dx + fy * dy + fz * fz <= 0 for dx, dy, dz in [0,1]

            // fx + fz + fz <= 0 for monotonicity along xyz diagonal
            bool xyzMonotone = 
                f000 >= (f100 + f010 + f001)/3.0 &&
                f111 <= (f011 + f101 + f110)/3.0 &&
                f000 + f100 >= f101 + f110 && 
                f000 + f010 >= f011 + f110 && 
                f000 + f001 >= f011 + f101 && 
                f111 + f011 <= f010 + f001 && 
                f111 + f101 <= f100 + f001 && 
                f111 + f110 <= f100 + f010;

            // fx + fy <= 0 for monotonicity along xy diagonal
            bool xyMonotone =
                f101 <= (f100 + f001)/2.0 &&
                f110 <= (f100 + f010)/2.0 &&
                f111 <= (f101 + f011)/2.0 &&
                f111 <= (f110 + f011)/2.0;

            // fx + fz <= 0 for monotonicity along xz plane
            bool xzMonotone =
                f000 >= (f001 + f100)/2.0 &&
                f000 >= (f010 + f100)/2.0 &&
                f001 >= (f011 + f101)/2.0 &&
                f010 >= (f011 + f110)/2.0;

            // fy + fz <= 0 for monotonicity along yz plane
            bool yzMonotone = 
                f001 <= (f000 + f010) / 2.0 &&
                f011 <= (f010 + f110) / 2.0 &&
                f011 <= (f010 + f110) / 2.0 &&
                f001 <= (f000 + f110) / 2.0;

            // fx <= 0 for monotonicity along x axis
            bool xMonotone =
                f100 <= f000 &&
                f110 <= f010 && 
                f101 <= f001 && 
                f111 <= f011;

            // fy <= 0 for monotonicity along y axis
            bool yMonotone = 
                f010 <= f000 && 
                f110 <= f100 && 
                f011 <= f001 && 
                f111 <= f101;

            // fz <= 0 for monotonicity along z axis
            bool zMonotone = 
                f001 <= f000 && 
                f101 <= f100 && 
                f011 <= f010 && 
                f111 <= f110;
              
            bool xDominantMonotone = xMonotone && xyMonotone && xzMonotone && xyzMonotone;
            bool yDominantMonotone = yMonotone && xyMonotone && yzMonotone && xyzMonotone;
            bool zDominantMonotone = zMonotone && xzMonotone && yzMonotone && xyzMonotone;

            return xDominantMonotone;
        } 

        bool getCellOcclusion2(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            float f000 = getVoxelSample(voxelCoords + ivec3(0, 0, 0));
            float f100 = getVoxelSample(voxelCoords + ivec3(1, 0, 0));
            float f010 = getVoxelSample(voxelCoords + ivec3(0, 1, 0));
            float f001 = getVoxelSample(voxelCoords + ivec3(0, 0, 1));
            float f011 = getVoxelSample(voxelCoords + ivec3(0, 1, 1));
            float f101 = getVoxelSample(voxelCoords + ivec3(1, 0, 1));
            float f110 = getVoxelSample(voxelCoords + ivec3(1, 1, 0));
            float f111 = getVoxelSample(voxelCoords + ivec3(1, 1, 1));

            bool xMonotone =
                f100 <= f000 &&
                f110 <= f010 && 
                f101 <= f001 && 
                f111 <= f011;

            bool yMonotone =
                f010 <= f000 &&
                f110 <= f100 &&
                f011 <= f001 &&
                f111 <= f101;

            bool zMonotone =
                f001 <= f000 &&
                f101 <= f100 &&
                f011 <= f010 &&
                f111 <= f110;

            bool monotone = xMonotone && yMonotone && zMonotone;

            // If not monotone, the cell may create interior maxima 
            return monotone;
        }


        float getCellMinOutputIncrementX(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            float f000 = getVoxelSample(voxelCoords + ivec3(0,0,0));
            float f100 = getVoxelSample(voxelCoords + ivec3(1,0,0));
            float f010 = getVoxelSample(voxelCoords + ivec3(0,1,0));
            float f001 = getVoxelSample(voxelCoords + ivec3(0,0,1));
            float f011 = getVoxelSample(voxelCoords + ivec3(0,1,1));
            float f101 = getVoxelSample(voxelCoords + ivec3(1,0,1));
            float f110 = getVoxelSample(voxelCoords + ivec3(1,1,0));
            float f111 = getVoxelSample(voxelCoords + ivec3(1,1,1));

            float v = INFINITY;

            v = min(v, (f000 - f111));
            v = min(v, (f010 - f111));
            v = min(v, (f001 - f111));
            v = min(v, (f011 - f111));
            v = min(v, (f000 - f101));
            v = min(v, (f001 - f101));
            v = min(v, (f000 - f110));
            v = min(v, (f010 - f110));
            v = min(v, (f000 - f100));

            v = max(v, 0.0);

            return v;
        }

        float getCellMinOutputIncrementY(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            float f000 = getVoxelSample(voxelCoords + ivec3(0,0,0));
            float f100 = getVoxelSample(voxelCoords + ivec3(1,0,0));
            float f010 = getVoxelSample(voxelCoords + ivec3(0,1,0));
            float f001 = getVoxelSample(voxelCoords + ivec3(0,0,1));
            float f011 = getVoxelSample(voxelCoords + ivec3(0,1,1));
            float f101 = getVoxelSample(voxelCoords + ivec3(1,0,1));
            float f110 = getVoxelSample(voxelCoords + ivec3(1,1,0));
            float f111 = getVoxelSample(voxelCoords + ivec3(1,1,1));

            float v = INFINITY;

            v = min(v, (f000 - f111));
            v = min(v, (f010 - f111));
            v = min(v, (f001 - f111));
            v = min(v, (f011 - f111));
            v = min(v, (f000 - f110));
            v = min(v, (f010 - f110));

            v = max(v, 0.0);

            return v;
        }

        float getCellMinOutputIncrementZ(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            float f000 = getVoxelSample(voxelCoords + ivec3(0,0,0));
            float f100 = getVoxelSample(voxelCoords + ivec3(1,0,0));
            float f010 = getVoxelSample(voxelCoords + ivec3(0,1,0));
            float f001 = getVoxelSample(voxelCoords + ivec3(0,0,1));
            float f011 = getVoxelSample(voxelCoords + ivec3(0,1,1));
            float f101 = getVoxelSample(voxelCoords + ivec3(1,0,1));
            float f110 = getVoxelSample(voxelCoords + ivec3(1,1,0));
            float f111 = getVoxelSample(voxelCoords + ivec3(1,1,1));

            float v = INFINITY;

            v = min(v, (f000 - f111));
            v = min(v, (f010 - f111));
            v = min(v, (f001 - f111));
            v = min(v, (f011 - f111));
            v = min(v, (f000 - f101));
            v = min(v, (f001 - f101));

            v = max(v, 0.0);

            return v;
        }

        float getCellMaxInputIncrementX(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords-1;

            float f000 = getVoxelSample(voxelCoords + ivec3(0,0,0));
            float f100 = getVoxelSample(voxelCoords + ivec3(1,0,0));
            float f010 = getVoxelSample(voxelCoords + ivec3(0,1,0));
            float f001 = getVoxelSample(voxelCoords + ivec3(0,0,1));
            float f011 = getVoxelSample(voxelCoords + ivec3(0,1,1));
            float f101 = getVoxelSample(voxelCoords + ivec3(1,0,1));
            float f110 = getVoxelSample(voxelCoords + ivec3(1,1,0));
            float f111 = getVoxelSample(voxelCoords + ivec3(1,1,1));

            float v = -INFINITY;

            v = max(v, (f100 - f000));
            v = max(v, (f110 - f000));
            v = max(v, (f101 - f000));
            v = max(v, (f111 - f000));
            v = max(v, (f110 - f010));
            v = max(v, (f111 - f010));
            v = max(v, (f101 - f001));
            v = max(v, (f111 - f001));
            v = max(v, (f111 - f011));
            v = max(v, (f000 + f010 + f100) / 3.0 - f000);
            v = max(v, (f000 + f001 + f100) / 3.0 - f000);
            v = max(v, (f010 + f100 + f110) / 3.0 - f000);
            v = max(v, (f001 + f100 + f101) / 3.0 - f000);
            v = max(v, (f001 + f010 + f100) / 3.0 - f000);
            v = max(v, (f011 + f101 + f110) / 3.0 - f000);
            v = max(v, (f011 + f110 + f111) / 3.0 - f010);
            v = max(v, (f011 + f110 + f010) / 3.0 - f010);
            v = max(v, (f011 + f101 + f111) / 3.0 - f001);
            v = max(v, (f011 + f101 + f001) / 3.0 - f001);
            v = max(v, 0.0);

            return v;
        }

        float getCellMaxInputIncrementY(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords-1;

            float f000 = getVoxelSample(voxelCoords + ivec3(0,0,0));
            float f100 = getVoxelSample(voxelCoords + ivec3(1,0,0));
            float f010 = getVoxelSample(voxelCoords + ivec3(0,1,0));
            float f001 = getVoxelSample(voxelCoords + ivec3(0,0,1));
            float f011 = getVoxelSample(voxelCoords + ivec3(0,1,1));
            float f101 = getVoxelSample(voxelCoords + ivec3(1,0,1));
            float f110 = getVoxelSample(voxelCoords + ivec3(1,1,0));
            float f111 = getVoxelSample(voxelCoords + ivec3(1,1,1));

            float v = -INFINITY;

            v = max(v, (f100 - f000));
            v = max(v, (f110 - f000));
            v = max(v, (f101 - f000));
            v = max(v, (f111 - f000));
            v = max(v, (f101 - f001));
            v = max(v, (f111 - f001));
            v = max(v, (f001 + f010 + f100) / 3.0 - f000);
            v = max(v, (f010 + f100 + f000) / 3.0 - f000);
            v = max(v, (f001 + f100 + f000) / 3.0 - f000);
            v = max(v, (f010 + f100 + f110) / 3.0 - f000);
            v = max(v, (f001 + f100 + f101) / 3.0 - f000);
            v = max(v, (f011 + f101 + f110) / 3.0 - f000);
            v = max(v, (f011 + f101 + f111) / 3.0 - f001);
            v = max(v, (f011 + f101 + f001) / 3.0 - f001);

            v = max(v, 0.0);

            return v;
        }

        float getCellMaxInputIncrementZ(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords-1;

            float f000 = getVoxelSample(voxelCoords + ivec3(0,0,0));
            float f100 = getVoxelSample(voxelCoords + ivec3(1,0,0));
            float f010 = getVoxelSample(voxelCoords + ivec3(0,1,0));
            float f001 = getVoxelSample(voxelCoords + ivec3(0,0,1));
            float f011 = getVoxelSample(voxelCoords + ivec3(0,1,1));
            float f101 = getVoxelSample(voxelCoords + ivec3(1,0,1));
            float f110 = getVoxelSample(voxelCoords + ivec3(1,1,0));
            float f111 = getVoxelSample(voxelCoords + ivec3(1,1,1));

            float v = -INFINITY;

            v = max(v, (f100 - f000));
            v = max(v, (f110 - f000));
            v = max(v, (f101 - f000));
            v = max(v, (f111 - f000));
            v = max(v, (f110 - f010));
            v = max(v, (f111 - f010));
            v = max(v, (f001 + f010 + f100) / 3.0 - f000);
            v = max(v, (f010 + f100 + f110) / 3.0 - f000);
            v = max(v, (f001 + f100 + f101) / 3.0 - f000);
            v = max(v, (f011 + f101 + f110) / 3.0 - f000);
            v = max(v, (f010 + f100 + f000) / 3.0 - f000);
            v = max(v, (f001 + f100 + f000) / 3.0 - f000);
            v = max(v, (f011 + f110 + f111) / 3.0 - f010);
            v = max(v, (f011 + f110 + f010) / 3.0 - f010);

            v = max(v, 0.0);

            return v;
        }

        vec3 getCellMaxInputIncrements(ivec3 cellCoords)
        {
            CellSamples C = getCellSamples(cellCoords);

            vec3 V = vec3(-INFINITY);

            V.x = max(V.x, (C.f100 - C.f000));
            V.x = max(V.x, (C.f110 - C.f000));
            V.x = max(V.x, (C.f101 - C.f000));
            V.x = max(V.x, (C.f111 - C.f000));
            V.x = max(V.x, (C.f110 - C.f010));
            V.x = max(V.x, (C.f111 - C.f010));
            V.x = max(V.x, (C.f101 - C.f001));
            V.x = max(V.x, (C.f111 - C.f001));
            V.x = max(V.x, (C.f111 - C.f011));
            V.x = max(V.x, (C.f000 + C.f010 + C.f100) / 3.0 - C.f000);
            V.x = max(V.x, (C.f000 + C.f001 + C.f100) / 3.0 - C.f000);
            V.x = max(V.x, (C.f010 + C.f100 + C.f110) / 3.0 - C.f000);
            V.x = max(V.x, (C.f001 + C.f100 + C.f101) / 3.0 - C.f000);
            V.x = max(V.x, (C.f001 + C.f010 + C.f100) / 3.0 - C.f000);
            V.x = max(V.x, (C.f011 + C.f101 + C.f110) / 3.0 - C.f000);
            V.x = max(V.x, (C.f011 + C.f110 + C.f111) / 3.0 - C.f010);
            V.x = max(V.x, (C.f011 + C.f110 + C.f010) / 3.0 - C.f010);
            V.x = max(V.x, (C.f011 + C.f101 + C.f111) / 3.0 - C.f001);
            V.x = max(V.x, (C.f011 + C.f101 + C.f001) / 3.0 - C.f001);

            V.y = max(V.y, (C.f100 - C.f000));
            V.y = max(V.y, (C.f110 - C.f000));
            V.y = max(V.y, (C.f101 - C.f000));
            V.y = max(V.y, (C.f111 - C.f000));
            V.y = max(V.y, (C.f101 - C.f001));
            V.y = max(V.y, (C.f111 - C.f001));
            V.y = max(V.y, (C.f001 + C.f010 + C.f100) / 3.0 - C.f000);
            V.y = max(V.y, (C.f010 + C.f100 + C.f000) / 3.0 - C.f000);
            V.y = max(V.y, (C.f001 + C.f100 + C.f000) / 3.0 - C.f000);
            V.y = max(V.y, (C.f010 + C.f100 + C.f110) / 3.0 - C.f000);
            V.y = max(V.y, (C.f001 + C.f100 + C.f101) / 3.0 - C.f000);
            V.y = max(V.y, (C.f011 + C.f101 + C.f110) / 3.0 - C.f000);
            V.y = max(V.y, (C.f011 + C.f101 + C.f111) / 3.0 - C.f001);
            V.y = max(V.y, (C.f011 + C.f101 + C.f001) / 3.0 - C.f001);

            V.z = max(V.z, (C.f100 - C.f000));
            V.z = max(V.z, (C.f110 - C.f000));
            V.z = max(V.z, (C.f101 - C.f000));
            V.z = max(V.z, (C.f111 - C.f000));
            V.z = max(V.z, (C.f110 - C.f010));
            V.z = max(V.z, (C.f111 - C.f010));
            V.z = max(V.z, (C.f001 + C.f010 + C.f100) / 3.0 - C.f000);
            V.z = max(V.z, (C.f010 + C.f100 + C.f110) / 3.0 - C.f000);
            V.z = max(V.z, (C.f001 + C.f100 + C.f101) / 3.0 - C.f000);
            V.z = max(V.z, (C.f011 + C.f101 + C.f110) / 3.0 - C.f000);
            V.z = max(V.z, (C.f010 + C.f100 + C.f000) / 3.0 - C.f000);
            V.z = max(V.z, (C.f001 + C.f100 + C.f000) / 3.0 - C.f000);
            V.z = max(V.z, (C.f011 + C.f110 + C.f111) / 3.0 - C.f010);
            V.z = max(V.z, (C.f011 + C.f110 + C.f010) / 3.0 - C.f010);

            V = max(V, 0.0);

            return V;
        }


        void main()
        {
            ivec3 outputCoords = getOutputCoords();
            ivec3 cellCoords = outputCoords.zyx;
            
            bool occluded = 
                getCellMaxInputIncrementX(cellCoords) <= 0.0 &&
                getCellMaxInputIncrementY(cellCoords) <= 0.0 &&
                getCellMaxInputIncrementZ(cellCoords) <= 0.0;

            setOutput(vec4(occluded));
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

export function computeOcclusionMap(interpolationMap: tf.Tensor3D) : tf.Tensor
{
    const program = new GPGPUOcclusionMap(interpolationMap.shape)
    return runProgram(program, [interpolationMap]) as tf.Tensor3D
}