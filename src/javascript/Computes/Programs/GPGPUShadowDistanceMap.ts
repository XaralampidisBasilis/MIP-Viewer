import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import {
    type Axis,
    type Dimension,
    type Octant,
    type Sign,
    type Tuple,
    axisIndex,
    mapFromDominantAxisOctant,
    reverseOctant,
    reverseSign,
    signFromOctant,
} from '../../Utils/ShadowMapUtils'

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

function logTensor(tag: string, tensor: tf.Tensor)
{
    console.log(tag, tf.tidy(() => tensor.mean([0,1,2]).dataSync())) 
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

function isotropicDistanceMapInt32(
    shadows: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    maxDistance: number,
    verbose: boolean = false,
): Int32Array
{
    const shape = shadows.shape as [number, number, number]
    const mapIndex = mapFromDominantAxisOctant(dominantAxis, octant)

    const pass0 = new ShadowChebyshevDistancePass(shape, mapIndex, maxDistance)
    const t0 = runWebGLProgram(pass0, [shadows], 'int32', [], false)

    const pass1 = new IsotropicChebyshevDistancePass(shape, 'x', maxDistance)
    const t1 = runWebGLProgram(pass1, [t0], 'int32', [], false)
    tf.dispose(t0)
    const pass2 = new IsotropicChebyshevDistancePass(shape, 'y', maxDistance)
    const t2 = runWebGLProgram(pass2, [t1], 'int32', [], false)
    tf.dispose(t1)
    const pass3 = new IsotropicChebyshevDistancePass(shape, 'z', maxDistance)
    const t3 = runWebGLProgram(pass3, [t2], 'int32', [], false)
    tf.dispose(t2)
    
    if (verbose) logTensor('distance', t3)

    const d = t3.dataSync() 
    tf.dispose(t3)

    return d as Int32Array
}

function extendedAnisotropicUnidirectionalDistanceMapInt32(
    shadows: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    maxDistance: number,
    verbose: boolean = false,
): Int32Array
{
    const axes = ['x', 'y', 'z'] as [Axis, Axis, Axis]
    const shape = shadows.shape as [number, number, number]

    const mapIndex = mapFromDominantAxisOctant(dominantAxis, octant)
    const dominantSign = signFromOctant(octant, axisIndex(dominantAxis))

    const otherAxes = axes.filter(a => a !== dominantAxis) as [Axis, Axis]
    const otherSigns = otherAxes.map(a => signFromOctant(octant, axisIndex(a))) as [Sign, Sign]

    const inAxes = [...otherAxes, dominantAxis] as [Axis, Axis, Axis]
    const inSigns = [...otherSigns, dominantSign] as [Sign, Sign, Sign]

    // initial distance
    const pass0 = new ShadowChebyshevDistancePass(shape, mapIndex, maxDistance)
    const t0 = runWebGLProgram(pass0, [shadows], 'int32', [], false)

    // forward distance
    const pass1 = new AnisotropicChebyshevDistancePass(shape, inAxes[0], inSigns[0], maxDistance)
    const t1 = runWebGLProgram(pass1, [t0], 'int32', [], false)
    tf.dispose(t0)
    const pass2 = new AnisotropicChebyshevDistancePass(shape, inAxes[1], inSigns[1], maxDistance)
    const t2 = runWebGLProgram(pass2, [t1], 'int32', [], false)
    tf.dispose(t1)
    const pass3 = new ExtendedChebyshevDistancePass(shape, inAxes[2], inSigns[2], maxDistance)
    const t3 = runWebGLProgram(pass3, [t2], 'int32', [], false)
    tf.dispose(t2)

    if (verbose) logTensor('distance', t3)

    const d = t3.dataSync() 
    tf.dispose(t3)

    return d as Int32Array
}

function extendedAnisotropicBidirectionalDistanceMapInt32(
    shadows: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    maxDistance: number,
    verbose: boolean = false,
): Int32Array
{
    const axes = ['x', 'y', 'z'] as [Axis, Axis, Axis]
    const shape = shadows.shape as [number, number, number]

    const mapIndex = mapFromDominantAxisOctant(dominantAxis, octant)
    const dominantSign = signFromOctant(octant, axisIndex(dominantAxis))

    const otherAxes = axes.filter(a => a !== dominantAxis) as [Axis, Axis]
    const otherSigns = otherAxes.map(a => signFromOctant(octant, axisIndex(a))) as [Sign, Sign]

    const inAxes = [...otherAxes, dominantAxis] as [Axis, Axis, Axis]
    const inSigns = [...otherSigns, dominantSign] as [Sign, Sign, Sign]
    const invSigns = inSigns.map((s) => reverseSign(s))

    // initial distance
    const pass0 = new ShadowChebyshevDistancePass(shape, mapIndex, maxDistance)
    const t0 = runWebGLProgram(pass0, [shadows], 'int32', [], false)

    // forward distance
    const pass1 = new AnisotropicChebyshevDistancePass(shape, inAxes[0], inSigns[0], maxDistance)
    const t1 = runWebGLProgram(pass1, [t0], 'int32', [], false)
    tf.dispose()
    const pass2 = new AnisotropicChebyshevDistancePass(shape, inAxes[1], inSigns[1], maxDistance)
    const t2 = runWebGLProgram(pass2, [t1], 'int32', [], false)
    tf.dispose(t1)
    const pass3 = new ExtendedChebyshevDistancePass(shape, inAxes[2], inSigns[2], maxDistance)
    const t3 = runWebGLProgram(pass3, [t2], 'int32', [], false)
    tf.dispose(t2)

    // backward distance
    const pass4 = new AnisotropicChebyshevDistancePass(shape, inAxes[0], invSigns[0], maxDistance)
    const t4 = runWebGLProgram(pass4, [t0], 'int32', [], false)
    tf.dispose(t0)
    const pass5 = new AnisotropicChebyshevDistancePass(shape, inAxes[1], invSigns[1], maxDistance)
    const t5 = runWebGLProgram(pass5, [t4], 'int32', [], false)
    tf.dispose(t4)
    const pass6 = new ExtendedChebyshevDistancePass(shape, inAxes[2], invSigns[2], maxDistance)
    const t6 = runWebGLProgram(pass6, [t5], 'int32', [], false)
    tf.dispose(t5)

    // bidirectional distance
    const t7 = tf.minimum(t3, t6)
    tf.dispose([t3, t6])

    if (verbose) logTensor('distance', t7)

    // sync distance
    const d = t7.dataSync() 
    tf.dispose(t7)

    return d as Int32Array
}

function packToR16UI(maps: Tuple<Int32Array, 12>): Uint16Array 
{
    const voxels = maps[0].length

    for (let i = 1; i < 12; i++) 
    {
        if (maps[i].length !== voxels) 
            throw new Error(`packToR16UI expected all maps to have length ${voxels}, got ${maps[i].length} at index ${i}`)
    }

    // RGBA16UI => 4 uint16 values per voxel
    const packed = new Uint16Array(voxels * 4)

    for (let i = 0; i < voxels; i++) 
    {
        // Clamp values to 5 bits
        const x0 = maps[ 0][i] & 0x1
        const x1 = maps[ 1][i] & 0x1
        const x2 = maps[ 2][i] & 0x1
        const x3 = maps[ 3][i] & 0x1

        // Clamp values to 5 bits
        const y0 = maps[ 4][i] & 0x1
        const y1 = maps[ 5][i] & 0x1
        const y2 = maps[ 6][i] & 0x1
        const y3 = maps[ 7][i] & 0x1

        // Clamp values to 6 bits
        const z0 = maps[ 8][i] & 0x1
        const z1 = maps[ 9][i] & 0x1
        const z2 = maps[10][i] & 0x1
        const z3 = maps[11][i] & 0x1

        // pack 12-bits into one uint16
        const r = 
        (x0 << 0) | (x1 << 1) | (x2 <<  2) | (x2 <<  3) |
        (y0 << 4) | (y1 << 5) | (y2 <<  6) | (y2 <<  7) |
        (z0 << 8) | (z1 << 9) | (z2 << 10) | (z2 << 11)

        packed[i] = r >>> 0

    }

    return packed
}

function packToRGBA16UI(maps: Tuple<Int32Array, 12>): Uint16Array 
{
    const voxels = maps[0].length

    for (let i = 1; i < 12; i++) 
    {
        if (maps[i].length !== voxels) 
            throw new Error(`packToRGBA16UI expected all maps to have length ${voxels}, got ${maps[i].length} at index ${i}`)
    }

    // RGBA16UI => 4 uint16 values per voxel
    const packed = new Uint16Array(voxels * 4)

    for (let i = 0; i < voxels; i++) 
    {
        // Clamp values to 5 bits
        const x0 = maps[ 0][i] & 0x1f
        const x1 = maps[ 1][i] & 0x1f
        const x2 = maps[ 2][i] & 0x1f
        const x3 = maps[ 3][i] & 0x1f

        // Clamp values to 5 bits
        const y0 = maps[ 4][i] & 0x1f
        const y1 = maps[ 5][i] & 0x1f
        const y2 = maps[ 6][i] & 0x1f
        const y3 = maps[ 7][i] & 0x1f

        // Clamp values to 6 bits
        const z0 = maps[ 8][i] & 0x3f
        const z1 = maps[ 9][i] & 0x3f
        const z2 = maps[10][i] & 0x3f
        const z3 = maps[11][i] & 0x3f

        // pack (5,5,6)-bit values into one uint16
        const r = (x0 << 0) | (y0 << 5) | (z0 << 10)
        const g = (x1 << 0) | (y1 << 5) | (z1 << 10)
        const b = (x2 << 0) | (y2 << 5) | (z2 << 10)
        const a = (x3 << 0) | (y3 << 5) | (z3 << 10)

        const i4 = i * 4
        packed[i4 + 0] = r >>> 0
        packed[i4 + 1] = g >>> 0
        packed[i4 + 2] = b >>> 0
        packed[i4 + 3] = a >>> 0
    }

    return packed
}

function packToRGB32UI(maps: Tuple<Int32Array, 12>): Uint32Array 
{
    const voxels = maps[0].length

    for (let i = 1; i < 12; i++) 
    {
        if (maps[i].length !== voxels) 
            throw new Error(`packToRGB32UI expected all maps to have length ${voxels}, got ${maps[i].length} at index ${i}`)
    }

    // RGB32UI => 3 uint32 values per voxel
    const packed = new Uint32Array(voxels * 3)

    for (let i = 0; i < voxels; i++) 
    {
        // Clamp to 8 bits
        const x0 = maps[ 0][i] & 0xff
        const x1 = maps[ 1][i] & 0xff
        const x2 = maps[ 2][i] & 0xff
        const x3 = maps[ 3][i] & 0xff

        const y0 = maps[ 4][i] & 0xff
        const y1 = maps[ 5][i] & 0xff
        const y2 = maps[ 6][i] & 0xff
        const y3 = maps[ 7][i] & 0xff

        const z0 = maps[ 8][i] & 0xff
        const z1 = maps[ 9][i] & 0xff
        const z2 = maps[10][i] & 0xff
        const z3 = maps[11][i] & 0xff

        // Pack 4x8-bit values into one uint32 per axis
        const r = (x0 << 0) | (x1 << 8) | (x2 << 16) | (x3 << 24)
        const g = (y0 << 0) | (y1 << 8) | (y2 << 16) | (y3 << 24)
        const b = (z0 << 0) | (z1 << 8) | (z2 << 16) | (z3 << 24)

        const i3 = i * 3
        packed[i3 + 0] = r >>> 0
        packed[i3 + 1] = g >>> 0
        packed[i3 + 2] = b >>> 0
    }

    return packed
}

export function computeIsotropicDistanceMap_RGBA16UI(shadows: tf.Tensor3D, verbose: boolean = false) : Uint16Array
{
    const maps = new Array(12) 

    maps[ 0] = isotropicDistanceMapInt32(shadows, 'x', '+++', 31, verbose)
    maps[ 1] = isotropicDistanceMapInt32(shadows, 'x', '+-+', 31, verbose)
    maps[ 2] = isotropicDistanceMapInt32(shadows, 'x', '++-', 31, verbose)
    maps[ 3] = isotropicDistanceMapInt32(shadows, 'x', '+--', 31, verbose)
    maps[ 4] = isotropicDistanceMapInt32(shadows, 'y', '+++', 31, verbose)
    maps[ 5] = isotropicDistanceMapInt32(shadows, 'y', '-++', 31, verbose)
    maps[ 6] = isotropicDistanceMapInt32(shadows, 'y', '++-', 31, verbose)
    maps[ 7] = isotropicDistanceMapInt32(shadows, 'y', '-+-', 31, verbose)
    maps[ 8] = isotropicDistanceMapInt32(shadows, 'z', '+++', 63, verbose)
    maps[ 9] = isotropicDistanceMapInt32(shadows, 'z', '+-+', 63, verbose)
    maps[10] = isotropicDistanceMapInt32(shadows, 'z', '-++', 63, verbose)
    maps[11] = isotropicDistanceMapInt32(shadows, 'z', '--+', 63, verbose)

    return packToRGBA16UI(maps as Tuple<Int32Array, 12>) 
}

export function computeIsotropicDistanceMap_RGB32UI(shadows: tf.Tensor3D, verbose: boolean = false) : Uint32Array
{
    const maps = new Array(12)

    maps[ 0] = isotropicDistanceMapInt32(shadows, 'x', '+++', 255, verbose)
    maps[ 1] = isotropicDistanceMapInt32(shadows, 'x', '+-+', 255, verbose)
    maps[ 2] = isotropicDistanceMapInt32(shadows, 'x', '++-', 255, verbose)
    maps[ 3] = isotropicDistanceMapInt32(shadows, 'x', '+--', 255, verbose)
    maps[ 4] = isotropicDistanceMapInt32(shadows, 'y', '+++', 255, verbose)
    maps[ 5] = isotropicDistanceMapInt32(shadows, 'y', '-++', 255, verbose)
    maps[ 6] = isotropicDistanceMapInt32(shadows, 'y', '++-', 255, verbose)
    maps[ 7] = isotropicDistanceMapInt32(shadows, 'y', '-+-', 255, verbose)
    maps[ 8] = isotropicDistanceMapInt32(shadows, 'z', '+++', 255, verbose)
    maps[ 9] = isotropicDistanceMapInt32(shadows, 'z', '+-+', 255, verbose)
    maps[10] = isotropicDistanceMapInt32(shadows, 'z', '-++', 255, verbose)
    maps[11] = isotropicDistanceMapInt32(shadows, 'z', '--+', 255, verbose)

    return packToRGB32UI(maps as Tuple<Int32Array, 12>) 
}

export function computeExtendedAnisotropicUnidirectionalDistanceMap_RGBA16UI(shadows: tf.Tensor3D, verbose: boolean = false) : Uint16Array
{
    const maps = new Array(12) 

    maps[ 0] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'x', '+++', 31, verbose)
    maps[ 1] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'x', '+-+', 31, verbose)
    maps[ 2] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'x', '++-', 31, verbose)
    maps[ 3] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'x', '+--', 31, verbose)
    maps[ 4] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'y', '+++', 31, verbose)
    maps[ 5] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'y', '-++', 31, verbose)
    maps[ 6] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'y', '++-', 31, verbose)
    maps[ 7] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'y', '-+-', 31, verbose)
    maps[ 8] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'z', '+++', 63, verbose)
    maps[ 9] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'z', '+-+', 63, verbose)
    maps[10] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'z', '-++', 63, verbose)
    maps[11] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'z', '--+', 63, verbose)

    return packToRGBA16UI(maps as Tuple<Int32Array, 12>) 
}

export function computeExtendedAnisotropicUnidirectionalDistanceMap_RGB32UI(shadows: tf.Tensor3D, verbose: boolean = false) : Uint32Array
{
    const maps = new Array(12)

    maps[ 0] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'x', '+++', 255, verbose)
    maps[ 1] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'x', '+-+', 255, verbose)
    maps[ 2] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'x', '++-', 255, verbose)
    maps[ 3] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'x', '+--', 255, verbose)
    maps[ 4] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'y', '+++', 255, verbose)
    maps[ 5] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'y', '-++', 255, verbose)
    maps[ 6] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'y', '++-', 255, verbose)
    maps[ 7] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'y', '-+-', 255, verbose)
    maps[ 8] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'z', '+++', 255, verbose)
    maps[ 9] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'z', '+-+', 255, verbose)
    maps[10] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'z', '-++', 255, verbose)
    maps[11] = extendedAnisotropicUnidirectionalDistanceMapInt32(shadows, 'z', '--+', 255, verbose)

    return packToRGB32UI(maps as Tuple<Int32Array, 12>) 
}

export function computeExtendedAnisotropicBidirectionalDistanceMap_RGBA16UI(shadows: tf.Tensor3D, verbose: boolean = false) : Uint16Array
{
    const maps = new Array(12) 

    maps[ 0] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'x', '+++', 31, verbose)
    maps[ 1] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'x', '+-+', 31, verbose)
    maps[ 2] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'x', '++-', 31, verbose)
    maps[ 3] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'x', '+--', 31, verbose)
    maps[ 4] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'y', '+++', 31, verbose)
    maps[ 5] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'y', '-++', 31, verbose)
    maps[ 6] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'y', '++-', 31, verbose)
    maps[ 7] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'y', '-+-', 31, verbose)
    maps[ 8] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'z', '+++', 63, verbose)
    maps[ 9] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'z', '+-+', 63, verbose)
    maps[10] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'z', '-++', 63, verbose)
    maps[11] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'z', '--+', 63, verbose)

    return packToRGBA16UI(maps as Tuple<Int32Array, 12>) 
}

export function computeExtendedAnisotropicBidirectionalDistanceMap_RGB32UI(shadows: tf.Tensor3D, verbose: boolean = false) : Uint32Array
{
    const maps = new Array(12)

    maps[ 0] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'x', '+++', 255, verbose)
    maps[ 1] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'x', '+-+', 255, verbose)
    maps[ 2] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'x', '++-', 255, verbose)
    maps[ 3] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'x', '+--', 255, verbose)
    maps[ 4] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'y', '+++', 255, verbose)
    maps[ 5] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'y', '-++', 255, verbose)
    maps[ 6] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'y', '++-', 255, verbose)
    maps[ 7] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'y', '-+-', 255, verbose)
    maps[ 8] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'z', '+++', 255, verbose)
    maps[ 9] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'z', '+-+', 255, verbose)
    maps[10] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'z', '-++', 255, verbose)
    maps[11] = extendedAnisotropicBidirectionalDistanceMapInt32(shadows, 'z', '--+', 255, verbose)

    return packToRGB32UI(maps as Tuple<Int32Array, 12>) 
}

