import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { packUnsignedShort5551 } from './GPGPUToUnsignedShort5551'

type Axis = 0 | 1 | 2
type Permute = [Axis, Axis, Axis]
type Reverse = Axis[]

class ShadowChebyshevDistancePass implements GPGPUProgram 
{
    variableNames = ['A'] 
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true
    customUniforms = [{ name: 'map', type: 'int' as const }]

    constructor(
        outputShape: [number, number, number],
        maxDistance: number = 31,
    ) {
        
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

            uvec4 outD = s * uvec4(${maxDistance});
            setOutput(vec4(outD));
        }
        `
    }
}

class AnisotropicChebyshevDistancePass implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        inputShape: [number, number, number],
        direction: '-x' | '+x' | '-y' | '+y' | '-z' | '+z',
        maxDistance: number = 31,
    ) {

        const sign = direction[0] as '-' | '+'
        const axis = direction[1] as 'x'|'y'|'z'

        const [D, H, W] = inputShape
        this.outputShape = [D, H, W]

        const maxX = W - 1
        const maxY = H - 1
        const maxZ = D - 1
        const stepDir = sign === '-' ? -1 : 1

        this.userCode = /* glsl */ `
        const ivec3 MAXC = ivec3(${maxX}, ${maxY}, ${maxZ});
        const int MAX_STEPS = min(${maxDistance}, MAXC.${axis});
        const int STEP_DIR = ${stepDir};

        int getAAt(ivec3 c)
        {
            return int(getA(c.z, c.y, c.x));
        }

        bool outOfBoundsAxis(int v)
        {
            return (v < 0) || (v > MAXC.${axis});
        }

        void main()
        {
            ivec3 outC = getOutputCoords().zyx;

            int outD = getAAt(outC);
            if (outD <= 1)
            {
                setOutput(float(outD));
                return;
            }

            ivec3 inC = outC;

            for (int n = 1; n <= MAX_STEPS; ++n)
            {
                int v = outC.${axis} + STEP_DIR * n;
                if (outOfBoundsAxis(v)) break;

                inC.${axis} = v;

                int inD = getAAt(inC);
                int varD = max(inD, n);

                outD = min(outD, varD);
                if (outD <= n) break;
            }

            setOutput(float(outD));
        }
        `
    }
}

class ExtendedChebyshevDistancePass implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        inputShape: [number, number, number],
        direction: '-x' | '+x' | '-y' | '+y' | '-z' | '+z',
        maxDistance: number = 31,
    ) {

        const sign = direction[0] as '-' | '+'           
        const axis = direction[1] as 'x'|'y'|'z'

        const [D, H, W] = inputShape
        this.outputShape = [D, H, W]

        const maxX = W - 1
        const maxY = H - 1
        const maxZ = D - 1
        const stepDir = sign === '-' ? -1 : 1

        this.userCode = /* glsl */ `
        const ivec3 MAXC = ivec3(${maxX}, ${maxY}, ${maxZ});
        const int MAX_STEPS = min(${maxDistance}, MAXC.${axis});
        const int STEP_DIR = ${stepDir};

        int getAAt(ivec3 c) 
        {
            return int(getA(c.z, c.y, c.x));
        }

        bool outOfBoundsAxis(int v) 
        {
            return (v < 0) || (v > MAXC.${axis});
        }

        void main() 
        {
            ivec3 outC = getOutputCoords().zyx;

            int d0 = getAAt(outC);
            if (d0 == 0) 
            {
                setOutput(0.0);
                return;
            }

            int best = ${maxDistance};
            ivec3 inC = outC;

            for (int n = 1; n <= MAX_STEPS; ++n) 
            {
                int v = outC.${axis} + STEP_DIR * n;
                if (outOfBoundsAxis(v)) break;

                inC.${axis} = v;

                int d = getAAt(inC);
                if (d <= n) 
                {
                    best = n;
                    break;
                }
            }

            setOutput(float(best));
        }
        `
    }
}

function runProgram(prog: GPGPUProgram, inputs: tf.Tensor[]) : tf.Tensor 
{
    const backend = tf.backend() as MathBackendWebGL
    const info = backend.compileAndRun(prog, inputs)
    return tf.engine().makeTensorFromTensorInfo(info) as tf.Tensor
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

export function computeExtendedAnisotropicBidirectionalShadowDistanceMap(shadowMap: tf.Tensor3D, maxDistance: number): tf.Tensor3D 
{
    const shape = shadowMap.shape as [number, number, number]

    const chebyshevPass = new ShadowChebyshevDistancePass(shape, maxDistance)
}
