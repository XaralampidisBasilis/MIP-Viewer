import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { packUnsignedShort5x3ToNormalizedHalfFloat } from './GPGPUToUnsignedShort5551'
import {
    type Axis,
    type Dimension,
    type Octant,
    type Sign,
    mapFromDominantAxisOctant,
    reverseOctant,
    reverseSign,
    signFromOctant,
} from './ShadowMapUtils'

type Array3<type> = [type, type, type]
type Array4<type> = [type, type, type, type]

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

class IsotropicChebyshevDistancePass implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        inputShape: [number, number, number],
        axis: Axis,
        maxDistance: number = 31,
    ) {
        const [D, H, W] = inputShape
        this.outputShape = [D, H, W]

        const maxX = W - 1
        const maxY = H - 1
        const maxZ = D - 1

        this.userCode = /* glsl */ `
        const ivec3 MAXC = ivec3(${maxX}, ${maxY}, ${maxZ});
        const int MAX_STEPS = min(${maxDistance}, MAXC.${axis});

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
                int v;

                v = outC.${axis} - n;
                if (!outOfBoundsAxis(v))
                {
                    inC.${axis} = v;

                    int inD = getAAt(inC);
                    int varD = max(inD, n);

                    outD = min(outD, varD);
                    if (outD <= n) break;
                }

                v = outC.${axis} + n;
                if (!outOfBoundsAxis(v))
                {
                    inC.${axis} = v;

                    int inD = getAAt(inC);
                    int varD = max(inD, n);

                    outD = min(outD, varD);
                    if (outD <= n) break;
                }
            }

            setOutput(float(outD));
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

class PackChannels implements GPGPUProgram 
{
    variableNames = ['R', 'G', 'B', 'A']
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = true

    constructor(inputShape: [number, number, number]) 
    {
        const [D, H, W] = inputShape
        this.outputShape = [D, H, W, 2, 2]

        this.userCode = /* glsl */ `
        ivec3 getOutCoords()
        {
            ivec5 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        float getRAt(ivec3 coords)
        {
            return getR(coords.z, coords.y, coords.x);
        }

        float getGAt(ivec3 coords)
        {
            return getG(coords.z, coords.y, coords.x);
        }

        float getBAt(ivec3 coords)
        {
            return getB(coords.z, coords.y, coords.x);
        }

        float getAAt(ivec3 coords)
        {
            return getA(coords.z, coords.y, coords.x);
        }
 
        void main()
        {
            ivec3 coords = getOutCoords();

            float r = getRAt(coords);
            float g = getGAt(coords);
            float b = getBAt(coords);
            float a = getAAt(coords);

            setOutput(vec4(r, g, b, a));
        }
        `
    }
}

export function computeShadowDistanceMapIsotropic(
    shadowMaps: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    maxDistance: number,
    verbose: boolean = false,
): tf.Tensor3D
{
    const shape = shadowMaps.shape as [number, number, number]
    const mapIndex = mapFromDominantAxisOctant(dominantAxis, octant)

    const pass0 = new ShadowChebyshevDistancePass(shape, mapIndex, maxDistance)
    const t0 = runWebGLProgram(pass0, [shadowMaps], 'float32', [], false)

    const pass1 = new IsotropicChebyshevDistancePass(shape, 'x', maxDistance)
    const t1 = runWebGLProgram(pass1, [t0], 'float32', [], false)
    tf.dispose(t0)
    const pass2 = new IsotropicChebyshevDistancePass(shape, 'y', maxDistance)
    const t2 = runWebGLProgram(pass2, [t1], 'float32', [], false)
    tf.dispose(t1)
    const pass3 = new IsotropicChebyshevDistancePass(shape, 'z', maxDistance)
    const t3 = runWebGLProgram(pass3, [t2], 'float32', [], false)
    tf.dispose(t2)

    if (verbose) logTensor(t3, 'IsotropicDistanceMap')

    return t3 as tf.Tensor3D
}

export function computeShadowDistanceMapExtendedAnisotropicUnidirectional(
    shadowMaps: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    maxDistance: number,
    verbose: boolean = false,
): tf.Tensor3D
{
    const axes = ['x', 'y', 'z'] as [Axis, Axis, Axis]
    const shape = shadowMaps.shape as [number, number, number]

    const mapIndex = mapFromDominantAxisOctant(dominantAxis, octant)
    const dominantSign = signFromOctant(octant, axisIndex(dominantAxis))

    const otherAxes = axes.filter(a => a !== dominantAxis) as [Axis, Axis]
    const otherSigns = otherAxes.map(a => signFromOctant(octant, axisIndex(a))) as [Sign, Sign]

    const inAxes = [...otherAxes, dominantAxis] as [Axis, Axis, Axis]
    const inSigns = [...otherSigns, dominantSign] as [Sign, Sign, Sign]

    const pass0 = new ShadowChebyshevDistancePass(shape, mapIndex, maxDistance)
    const t0 = runWebGLProgram(pass0, [shadowMaps], 'float32', [], false)

    const pass1 = new AnisotropicChebyshevDistancePass(shape, inAxes[0], inSigns[0], maxDistance)
    const t1 = runWebGLProgram(pass1, [t0], 'float32', [], false)
    tf.dispose(t0)
    const pass2 = new AnisotropicChebyshevDistancePass(shape, inAxes[1], inSigns[1], maxDistance)
    const t2 = runWebGLProgram(pass2, [t1], 'float32', [], false)
    tf.dispose(t1)
    const pass3 = new ExtendedChebyshevDistancePass(shape, inAxes[2], inSigns[2], maxDistance)
    const t3 = runWebGLProgram(pass3, [t2], 'float32', [], false)
    tf.dispose(t2)

    if (verbose) logTensor(t3, "distanceMap")

    return t3 as tf.Tensor3D
}

export function computeShadowDistanceMapExtendedAnisotropicBidirectional(
    shadowMaps: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    maxDistance: number,
    verbose: boolean = false,
): tf.Tensor3D
{
    const forwardDistanceMap = computeShadowDistanceMapExtendedAnisotropicUnidirectional(shadowMaps, dominantAxis, octant, maxDistance)
    const backwardDistanceMap = computeShadowDistanceMapExtendedAnisotropicUnidirectional(shadowMaps, dominantAxis, reverseOctant(octant), maxDistance)

    const distanceMap = tf.minimum(forwardDistanceMap, backwardDistanceMap)
    if (verbose) logTensor(distanceMap, "distanceMap")

    tf.dispose([forwardDistanceMap, backwardDistanceMap])

    return distanceMap as tf.Tensor3D
}

export function computeIsotropicDistanceMap(
    shadowMaps: tf.Tensor3D,
    maxDistance: number = 31,
    verbose: boolean = false
) : tf.Tensor5D
{
    let anisotropicMaps = []
    let extendedMaps = [] 

    extendedMaps[0] = computeShadowDistanceMapIsotropic(shadowMaps, 'x', '+++', maxDistance, verbose)
    extendedMaps[1] = computeShadowDistanceMapIsotropic(shadowMaps, 'y', '+++', maxDistance, verbose)
    extendedMaps[2] = computeShadowDistanceMapIsotropic(shadowMaps, 'z', '+++', maxDistance, verbose)

    anisotropicMaps[0] = packUnsignedShort5x3ToNormalizedHalfFloat(...extendedMaps as Array3<tf.Tensor3D>)
    tf.dispose(extendedMaps)

    extendedMaps[0] = computeShadowDistanceMapIsotropic(shadowMaps, 'x', '+-+', maxDistance, verbose)
    extendedMaps[1] = computeShadowDistanceMapIsotropic(shadowMaps, 'y', '-++', maxDistance, verbose)
    extendedMaps[2] = computeShadowDistanceMapIsotropic(shadowMaps, 'z', '+-+', maxDistance, verbose)

    anisotropicMaps[1] = packUnsignedShort5x3ToNormalizedHalfFloat(...extendedMaps as Array3<tf.Tensor3D>)
    tf.dispose(extendedMaps)

    extendedMaps[0] = computeShadowDistanceMapIsotropic(shadowMaps, 'x', '++-', maxDistance, verbose)
    extendedMaps[1] = computeShadowDistanceMapIsotropic(shadowMaps, 'y', '++-', maxDistance, verbose)
    extendedMaps[2] = computeShadowDistanceMapIsotropic(shadowMaps, 'z', '-++', maxDistance, verbose)

    anisotropicMaps[2] = packUnsignedShort5x3ToNormalizedHalfFloat(...extendedMaps as Array3<tf.Tensor3D>)
    tf.dispose(extendedMaps)

    extendedMaps[0] = computeShadowDistanceMapIsotropic(shadowMaps, 'x', '+--', maxDistance, verbose)
    extendedMaps[1] = computeShadowDistanceMapIsotropic(shadowMaps, 'y', '-+-', maxDistance, verbose)
    extendedMaps[2] = computeShadowDistanceMapIsotropic(shadowMaps, 'z', '--+', maxDistance, verbose)

    anisotropicMaps[3] = packUnsignedShort5x3ToNormalizedHalfFloat(...extendedMaps as Array3<tf.Tensor3D>)
    tf.dispose(extendedMaps)

    const pack = new PackChannels(shadowMaps.shape)
    const distanceMap = runWebGLProgram(pack, anisotropicMaps as Array4<tf.Tensor3D>, 'float32', [], true)

    tf.dispose(anisotropicMaps)

    if (verbose) 
    {
        logTensor(distanceMap, "ExtendedAnisotropicBidirectionalShadowDistanceMap")
    }

    return distanceMap as tf.Tensor5D
}

export function computeExtendedAnisotropicBidirectionalDistanceMap(
    shadowMaps: tf.Tensor3D,
    maxDistance: number = 31,
    verbose: boolean = false
) : tf.Tensor5D
{
    let anisotropicMaps = []
    let extendedMaps = [] 

    extendedMaps[0] = computeShadowDistanceMapExtendedAnisotropicBidirectional(shadowMaps, 'x', '+++', maxDistance, verbose)
    extendedMaps[1] = computeShadowDistanceMapExtendedAnisotropicBidirectional(shadowMaps, 'y', '+++', maxDistance, verbose)
    extendedMaps[2] = computeShadowDistanceMapExtendedAnisotropicBidirectional(shadowMaps, 'z', '+++', maxDistance, verbose)

    anisotropicMaps[0] = packUnsignedShort5x3ToNormalizedHalfFloat(...extendedMaps as Array3<tf.Tensor3D>)
    tf.dispose(extendedMaps)

    extendedMaps[0] = computeShadowDistanceMapExtendedAnisotropicBidirectional(shadowMaps, 'x', '+-+', maxDistance, verbose)
    extendedMaps[1] = computeShadowDistanceMapExtendedAnisotropicBidirectional(shadowMaps, 'y', '-++', maxDistance, verbose)
    extendedMaps[2] = computeShadowDistanceMapExtendedAnisotropicBidirectional(shadowMaps, 'z', '+-+', maxDistance, verbose)

    anisotropicMaps[1] = packUnsignedShort5x3ToNormalizedHalfFloat(...extendedMaps as Array3<tf.Tensor3D>)
    tf.dispose(extendedMaps)

    extendedMaps[0] = computeShadowDistanceMapExtendedAnisotropicBidirectional(shadowMaps, 'x', '++-', maxDistance, verbose)
    extendedMaps[1] = computeShadowDistanceMapExtendedAnisotropicBidirectional(shadowMaps, 'y', '++-', maxDistance, verbose)
    extendedMaps[2] = computeShadowDistanceMapExtendedAnisotropicBidirectional(shadowMaps, 'z', '-++', maxDistance, verbose)

    anisotropicMaps[2] = packUnsignedShort5x3ToNormalizedHalfFloat(...extendedMaps as Array3<tf.Tensor3D>)
    tf.dispose(extendedMaps)

    extendedMaps[0] = computeShadowDistanceMapExtendedAnisotropicBidirectional(shadowMaps, 'x', '+--', maxDistance, verbose)
    extendedMaps[1] = computeShadowDistanceMapExtendedAnisotropicBidirectional(shadowMaps, 'y', '-+-', maxDistance, verbose)
    extendedMaps[2] = computeShadowDistanceMapExtendedAnisotropicBidirectional(shadowMaps, 'z', '--+', maxDistance, verbose)

    anisotropicMaps[3] = packUnsignedShort5x3ToNormalizedHalfFloat(...extendedMaps as Array3<tf.Tensor3D>)
    tf.dispose(extendedMaps)

    const pack = new PackChannels(shadowMaps.shape)
    const distanceMap = runWebGLProgram(pack, anisotropicMaps as Array4<tf.Tensor3D>, 'float32', [], true)
    tf.dispose(anisotropicMaps)

    if (verbose) 
        logTensor(distanceMap, "ExtendedAnisotropicBidirectionalShadowDistanceMap")

    return distanceMap as tf.Tensor5D
}

export function computeExtendedAnisotropicBidirectionalDistanceMapDebug(
    shadowMaps: tf.Tensor3D,
    maxDistance: number = 31,
    verbose: boolean = false
) : tf.Tensor5D
{
    const t = computeShadowDistanceMapIsotropic(shadowMaps, 'z', '-++', maxDistance, verbose)
    
    let anisotropicMaps = []
    let extendedMaps = [] 

    extendedMaps[0] = tf.onesLike(t) // 'x', '+++'
    extendedMaps[1] = tf.onesLike(t) // 'y', '+++'
    extendedMaps[2] = tf.onesLike(t) // 'z', '+++'

    anisotropicMaps[0] = packUnsignedShort5x3ToNormalizedHalfFloat(...extendedMaps as Array3<tf.Tensor3D>)
    tf.dispose(extendedMaps)

    extendedMaps[0] = tf.onesLike(t) // 'x', '+-+'
    extendedMaps[1] = tf.onesLike(t) // 'y', '-++'
    extendedMaps[2] = tf.onesLike(t) // 'z', '+-+'

    anisotropicMaps[1] = packUnsignedShort5x3ToNormalizedHalfFloat(...extendedMaps as Array3<tf.Tensor3D>)
    tf.dispose(extendedMaps)

    extendedMaps[0] = tf.onesLike(t) // 'x', '++-'
    extendedMaps[1] = tf.onesLike(t) // 'y', '++-'
    extendedMaps[2] = tf.clone(t) // 'z', '-++',

    anisotropicMaps[2] = packUnsignedShort5x3ToNormalizedHalfFloat(...extendedMaps as Array3<tf.Tensor3D>)
    tf.dispose(extendedMaps)

    extendedMaps[0] = tf.onesLike(t) // 'x', '+--'
    extendedMaps[1] = tf.onesLike(t) // 'y', '-+-'
    extendedMaps[2] = tf.onesLike(t) // 'z', '--+'

    anisotropicMaps[3] = packUnsignedShort5x3ToNormalizedHalfFloat(...extendedMaps as Array3<tf.Tensor3D>)
    tf.dispose(extendedMaps)

    const pack = new PackChannels(shadowMaps.shape)
    const distanceMap = runWebGLProgram(pack, anisotropicMaps as Array4<tf.Tensor3D>, 'float32', [], true)

    tf.dispose(anisotropicMaps)

    if (verbose) 
    {
        logTensor(distanceMap, "ExtendedAnisotropicBidirectionalShadowDistanceMap")
    }

    tf.dispose(t)

    return distanceMap as tf.Tensor5D
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

