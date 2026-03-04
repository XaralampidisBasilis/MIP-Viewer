import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { packUnsignedShort5551 } from './GPGPUToUnsignedShort5551'
import {
    type Axis,
    type Dimension,
    type Octant,
    type Sign,
    mapFromDominantAxisOctant,
    reverseSign,
    signFromOctant,
} from './ShadowMapUtils'

class ShadowChebyshevDistancePass implements GPGPUProgram 
{
    variableNames = ['A'] 
    outputShape: number[]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        outputShape: [number, number, number],
        map: number,
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
            uvec4 s = (u >> ${map}) & 1u;

            uvec4 outD = uvec4(${maxDistance}) * s;
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
        axis: Axis,
        sign: Sign,
        maxDistance: number = 31,
    ) {
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
        axis: Axis,
        sign: Sign,        
        maxDistance: number = 31,
    ) {
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

export function computeExtendedAnisotropicBidirectionalShadowDistanceMap(
    shadowMaps: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    maxDistance: number
): tf.Tensor3D
{
    const shape = shadowMaps.shape as [number, number, number]
    const dominantSign = signFromOctant(octant, axisIndex(dominantAxis))

    const otherAxes = ['x', 'y', 'z'].filter(a => a !== dominantAxis) as [Axis, Axis]
    const otherSigns = otherAxes.map(a => signFromOctant(octant, axisIndex(a))) as [Sign, Sign]

    const mapIndex = mapFromDominantAxisOctant(dominantAxis, octant)

    const passes = [
        new ShadowChebyshevDistancePass(shape, mapIndex, maxDistance),

        new AnisotropicChebyshevDistancePass(shape, otherAxes[0], otherSigns[0], maxDistance),
        new AnisotropicChebyshevDistancePass(shape, otherAxes[1], otherSigns[1], maxDistance),
        new ExtendedChebyshevDistancePass(shape, dominantAxis, dominantSign, maxDistance),

        new AnisotropicChebyshevDistancePass(shape, otherAxes[0], reverseSign(otherSigns[0]), maxDistance),
        new AnisotropicChebyshevDistancePass(shape, otherAxes[1], reverseSign(otherSigns[1]), maxDistance),
        new ExtendedChebyshevDistancePass(shape, dominantAxis, reverseSign(dominantSign), maxDistance),
    ]

    let t: tf.Tensor = shadowMaps
    for (const pass of passes)
    {
        const out = runWebGLProgram(pass, [t as tf.Tensor3D], 'float32', [], false)
        if (t !== shadowMaps) tf.dispose(t)
        t = out
    }

    logTensor(t, "Extended Anisotropic Bidirectional Shadow Distance Map")

    return t as tf.Tensor3D
}

// helper functions

function logTensor(tensor: tf.Tensor, name: string)
{
    console.log(name, tf.tidy(() => tensor.mean([0,1,2]).dataSync())) 
}

function axisIndex(axis: Axis): Dimension
{
    const axes = ['x', 'y', 'z'] as Axis[]
    return axes.indexOf(axis as any) as Dimension
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

