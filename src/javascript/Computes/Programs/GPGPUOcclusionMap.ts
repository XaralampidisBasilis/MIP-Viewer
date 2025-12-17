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

        const ivec3 voxelMinCoords = ivec3(0);
        const ivec3 voxelMaxCoords = ivec3(${inWidth-1}, ${inHeight-1}, ${inDepth-1});

        struct Cell
        {
            float samples[8];
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

        Cell getCellSamples(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            Cell cell;
            cell.samples[0] = getVoxelSample(voxelCoords + ivec3(0,0,0));
            cell.samples[1] = getVoxelSample(voxelCoords + ivec3(1,0,0));
            cell.samples[2] = getVoxelSample(voxelCoords + ivec3(0,1,0));
            cell.samples[3] = getVoxelSample(voxelCoords + ivec3(0,0,1));
            cell.samples[4] = getVoxelSample(voxelCoords + ivec3(0,1,1));
            cell.samples[5] = getVoxelSample(voxelCoords + ivec3(1,0,1));
            cell.samples[6] = getVoxelSample(voxelCoords + ivec3(1,1,0));
            cell.samples[7] = getVoxelSample(voxelCoords + ivec3(1,1,1));

            return cell;
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

        bool getCellOcclusion3(ivec3 cellCoords)
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

            bool monotone = 
            f100 <= f000 &&         
            f110 <= f010 &&         
            f101 <= f001 &&         
            f111 <= f011 &&  
            f000 >= (f010 + f100) / 2.0 && 
            f000 >= (f001 + f100) / 2.0 &&
            f010 >= (f011 + f110) / 2.0 &&
            f001 >= (f011 + f101) / 2.0 &&
            f110 <= (f010 + f100) / 2.0 && 
            f101 <= (f001 + f100) / 2.0 &&
            f111 <= (f011 + f110) / 2.0 &&
            f111 <= (f011 + f101) / 2.0 &&
            f000 >= (f001 + f010 + f100) / 3.0 &&   
            f111 <= (f011 + f101 + f110) / 3.0 &&
            f011 + f101 + f110 <= f001 + f010 + f100;

            return monotone;
        } 

          float getCellMaxXIncrement(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            float f_000 = getVoxelSample(voxelCoords + ivec3(0,0,0));
            float f_100 = getVoxelSample(voxelCoords + ivec3(1,0,0));
            float f_010 = getVoxelSample(voxelCoords + ivec3(0,1,0));
            float f_001 = getVoxelSample(voxelCoords + ivec3(0,0,1));
            float f_011 = getVoxelSample(voxelCoords + ivec3(0,1,1));
            float f_101 = getVoxelSample(voxelCoords + ivec3(1,0,1));
            float f_110 = getVoxelSample(voxelCoords + ivec3(1,1,0));
            float f_111 = getVoxelSample(voxelCoords + ivec3(1,1,1));

            float s_100_010 = f_100 + f_010;
            float s_100_001 = f_100 + f_001;
            float s_011_101 = f_011 + f_101;
            float s_011_110 = f_011 + f_110;

            float s_100_010_001 = f_001 + f_010 + f_100;
            float s_011_101_110 = f_011 + f_101 + f_110;

            float m = -1.0/0.0;

            m = max(m, f_100 - f_000);
            m = max(m, f_110 - f_010);
            m = max(m, f_101 - f_001);
            m = max(m, f_111 - f_011);
            m = max(m, s_100_010 - 2.0 * f_000);
            m = max(m, s_100_001 - 2.0 * f_000);
            m = max(m, s_011_101 - 2.0 * f_001);
            m = max(m, s_011_110 - 2.0 * f_010);
            m = max(m, 2.0 * f_110 - s_100_010);
            m = max(m, 2.0 * f_101 - s_100_001);
            m = max(m, 2.0 * f_111 - s_011_101);
            m = max(m, 2.0 * f_111 - s_011_110);
            m = max(m, s_100_010_001 - 3.0 * f_000);
            m = max(m, 3.0 * f_111 - s_011_101_110);

            m = max(m, s_011_101_110 - s_100_010_001);

            m = max(m, 0.0);

            return m;
        }

        float getCellMinXIncrement(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            float f_000 = getVoxelSample(voxelCoords + ivec3(0,0,0));
            float f_100 = getVoxelSample(voxelCoords + ivec3(1,0,0));
            float f_010 = getVoxelSample(voxelCoords + ivec3(0,1,0));
            float f_001 = getVoxelSample(voxelCoords + ivec3(0,0,1));
            float f_011 = getVoxelSample(voxelCoords + ivec3(0,1,1));
            float f_101 = getVoxelSample(voxelCoords + ivec3(1,0,1));
            float f_110 = getVoxelSample(voxelCoords + ivec3(1,1,0));
            float f_111 = getVoxelSample(voxelCoords + ivec3(1,1,1));

            float s_100_010 = f_100 + f_010;
            float s_100_001 = f_100 + f_001;
            float s_011_101 = f_011 + f_101;
            float s_011_110 = f_011 + f_110;

            float s_100_010_001 = f_001 + f_010 + f_100;
            float s_011_101_110 = f_011 + f_101 + f_110;

            float m = 1.0/0.0;

            m = min(m, f_000 - f_100);
            m = min(m, f_010 - f_110);
            m = min(m, f_001 - f_101);
            m = min(m, f_011 - f_111);
            m = min(m, 2.0 * f_000 - s_100_010);
            m = min(m, 2.0 * f_000 - s_100_001);
            m = min(m, 2.0 * f_001 - s_011_101);
            m = min(m, 2.0 * f_010 - s_011_110);
            m = min(m, s_100_010 - 2.0 * f_110);
            m = min(m, s_100_001 - 2.0 * f_101);
            m = min(m, s_011_101 - 2.0 * f_111);
            m = min(m, s_011_110 - 2.0 * f_111);
            m = min(m, 3.0 * f_000  - s_100_010_001);
            m = min(m, s_011_101_110 - 3.0 * f_111);
            m = min(m, s_100_010_001 - s_011_101_110);

            m = max(m, 0.0);

            return m;
        }

        bool isCellConvex(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            float f_000 = getVoxelSample(voxelCoords + ivec3(0,0,0));
            float f_100 = getVoxelSample(voxelCoords + ivec3(1,0,0));
            float f_010 = getVoxelSample(voxelCoords + ivec3(0,1,0));
            float f_001 = getVoxelSample(voxelCoords + ivec3(0,0,1));
            float f_011 = getVoxelSample(voxelCoords + ivec3(0,1,1));
            float f_101 = getVoxelSample(voxelCoords + ivec3(1,0,1));
            float f_110 = getVoxelSample(voxelCoords + ivec3(1,1,0));
            float f_111 = getVoxelSample(voxelCoords + ivec3(1,1,1));

            float s_100_010_001 = f_001 + f_010 + f_100;
            float s_011_101_110 = f_011 + f_101 + f_110;

            float v_111 = f_000 - f_001 - f_010 + f_011; 
            float v_011 = f_000 - f_001 - f_100 + f_101;
            float v_101 = f_000 - f_010 - f_100 + f_110;
            float v_110 = s_100_010_001 - s_011_101_110 + f_111 - f_000;

            bool convex = 
            v_111 >= 0.0 &&
            min(min(v_101, v_101), v_011 + v_101 + v_110) >= 0.0;


            return convex;
        }

         float getCellMinXIncrement3(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            float f_000 = getVoxelSample(voxelCoords + ivec3(0,0,0));
            float f_100 = getVoxelSample(voxelCoords + ivec3(1,0,0));
            float f_010 = getVoxelSample(voxelCoords + ivec3(0,1,0));
            float f_001 = getVoxelSample(voxelCoords + ivec3(0,0,1));
            float f_011 = getVoxelSample(voxelCoords + ivec3(0,1,1));
            float f_101 = getVoxelSample(voxelCoords + ivec3(1,0,1));
            float f_110 = getVoxelSample(voxelCoords + ivec3(1,1,0));
            float f_111 = getVoxelSample(voxelCoords + ivec3(1,1,1));

            float m = 1.0/0.0;

            m = min(m, (f_000 - f_111));
            m = min(m, (f_010 - f_111));
            m = min(m, (f_001 - f_111));
            m = min(m, (f_011 - f_111));
            m = min(m, (f_000 - f_101));
            m = min(m, (f_001 - f_101));
            m = min(m, (f_000 - f_110));
            m = min(m, (f_010 - f_110));
            m = min(m, (f_000 - f_100));
            m = max(m, 0.0);

            return m;
        }

        float getCellMaxXIncrement3(ivec3 cellCoords)
        {
            ivec3 voxelCoords = cellCoords - 1;

            float f_000 = getVoxelSample(voxelCoords + ivec3(0,0,0));
            float f_100 = getVoxelSample(voxelCoords + ivec3(1,0,0));
            float f_010 = getVoxelSample(voxelCoords + ivec3(0,1,0));
            float f_001 = getVoxelSample(voxelCoords + ivec3(0,0,1));
            float f_011 = getVoxelSample(voxelCoords + ivec3(0,1,1));
            float f_101 = getVoxelSample(voxelCoords + ivec3(1,0,1));
            float f_110 = getVoxelSample(voxelCoords + ivec3(1,1,0));
            float f_111 = getVoxelSample(voxelCoords + ivec3(1,1,1));

            float m = -1.0/0.0;

            m = max(m, (f_100 - f_000));
            m = max(m, (f_110 - f_000));
            m = max(m, (f_101 - f_000));
            m = max(m, (f_111 - f_000));
            m = max(m, (f_110 - f_010));
            m = max(m, (f_111 - f_010));
            m = max(m, (f_101 - f_001));
            m = max(m, (f_111 - f_001));
            m = max(m, (f_111 - f_011));
            // m = max(m, (f_010 + f_100 - 2.0 * f_000) / 3.0);
            // m = max(m, (f_001 + f_100 - 2.0 * f_000) / 3.0);
            // m = max(m, (f_011 + f_110 - 2.0 * f_010) / 3.0);
            // m = max(m, (f_011 + f_101 - 2.0 * f_001) / 3.0);
            // m = max(m, (f_001 + f_010 + f_100) / 3.0 - f_000);
            // m = max(m, (f_010 + f_100 + f_110) / 3.0 - f_000);
            // m = max(m, (f_001 + f_100 + f_101) / 3.0 - f_000);
            // m = max(m, (f_011 + f_101 + f_110) / 3.0 - f_000);
            // m = max(m, (f_011 + f_110 + f_111) / 3.0 - f_010);
            // m = max(m, (f_011 + f_101 + f_111) / 3.0 - f_001);
            m = max(m, 0.0);

            return m;
        }

        void main()
        {
            ivec3 outputCoords = getOutputCoords();
            ivec3 cellCoords = outputCoords.zyx;
            
            bool cellOcclusion = getCellOcclusion(cellCoords);

            setOutput(vec4(getCellMaxXIncrement3(cellCoords)));
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