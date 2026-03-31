import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
// import { computeBidirectionalBlockShadowMap } from './GPGPUShadowMap'
import { computeBidirectionalBlockShadowMap } from './GPGPUShadowMapDifferences'
import {
    type Axis,
    type Octant,
    type Sign,
    type Tuple,
    axisIndex,
    reverseSign,
    signFromOctant,
} from '../../Utils/ShadowMapUtils'


class ShadowChebyshevDistancePass implements GPGPUProgram 
{
    variableNames = ['A'] 
    outputShape: number[]
    userCode: string
    packedInputs = false
    packedOutput = false

    constructor(
        outputShape: [number, number, number],
        maxDistance: number,
    ) {
        
        this.outputShape = outputShape
        this.userCode = `
        uint toUint(float a) 
        { 
            return uint(round(a)); 
        }

        ivec3 getOutCoords() 
        {
            ivec3 coords = getOutputCoords();
            return ivec3(coords.z, coords.y, coords.x);
        }

        float getAAt(ivec3 coords) 
        {
            return getA(coords.z, coords.y, coords.x);
        }

        void main() 
        {
            ivec3 coords = getOutCoords();
            uint s = toUint(getAAt(coords));

            uint outD = uint(${maxDistance}) * s;
            setOutput(float(outD));
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
        maxDistance: number,
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
        maxDistance: number,
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
        maxDistance: number,
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

function packToR16UIArray(maps: Tuple<Int32Array, 12>): Uint16Array 
{
    const voxels = maps[0].length

    for (let i = 1; i < 12; i++) 
    {
        if (maps[i].length !== voxels) 
            throw new Error(`packToR16UI expected all maps to have length ${voxels}, got ${maps[i].length} at index ${i}`)
    }

    // RGBA16UI => 4 uint16 values per voxel
    const packed = new Uint16Array(voxels)

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
        (x0 << 0) | (x1 << 1) | (x2 <<  2) | (x3 <<  3) |
        (y0 << 4) | (y1 << 5) | (y2 <<  6) | (y3 <<  7) |
        (z0 << 8) | (z1 << 9) | (z2 << 10) | (z3 << 11)

        packed[i] = r >>> 0

    }

    return packed
}

function packToRGBA16UIArray(maps: Tuple<Int32Array, 12>): Uint16Array 
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

function packToRGB32UIArray(maps: Tuple<Int32Array, 12>): Uint32Array 
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

function packToRGBA32UIArray(maps: Tuple<Int32Array, 12>): Uint32Array 
{
    const voxels = maps[0].length

    for (let i = 1; i < 12; i++) 
    {
        if (maps[i].length !== voxels) 
            throw new Error(`packToRGBA32UIArray expected all maps to have length ${voxels}, got ${maps[i].length} at index ${i}`)
    }

    // RGBA16UI => 4 uint16 values per voxel
    const packed = new Uint32Array(voxels * 4)

    for (let i = 0; i < voxels; i++) 
    {
        // Clamp values to 5 bits
        const x0 = maps[ 0][i] & 0x7ff
        const x1 = maps[ 1][i] & 0x7ff
        const x2 = maps[ 2][i] & 0x7ff
        const x3 = maps[ 3][i] & 0x7ff

        // Clamp values to 5 bits0x3ff
        const y0 = maps[ 4][i] & 0x7ff
        const y1 = maps[ 5][i] & 0x7ff
        const y2 = maps[ 6][i] & 0x7ff
        const y3 = maps[ 7][i] & 0x7ff

        // Clamp values to 6 bits0x3ff
        const z0 = maps[ 8][i] & 0x3ff
        const z1 = maps[ 9][i] & 0x3ff
        const z2 = maps[10][i] & 0x3ff
        const z3 = maps[11][i] & 0x3ff

        // pack (5,5,6)-bit values into one uint16
        const r = (x0 << 0) | (y0 << 11) | (z0 << 22)
        const g = (x1 << 0) | (y1 << 11) | (z1 << 22)
        const b = (x2 << 0) | (y2 << 11) | (z2 << 22)
        const a = (x3 << 0) | (y3 << 11) | (z3 << 22)

        const i4 = i * 4
        packed[i4 + 0] = r >>> 0
        packed[i4 + 1] = g >>> 0
        packed[i4 + 2] = b >>> 0
        packed[i4 + 3] = a >>> 0
    }

    return packed
}

function isotropicDistanceMapInt32Array(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    tolerance: number, 
    blockSize: number,
    maxDistance: number,
    verbose: boolean = false,
): Int32Array
{
    const shadows = computeBidirectionalBlockShadowMap(volume, dominantAxis, octant, tolerance, blockSize, false)
    const shape = shadows.shape as [number, number, number]

    if (verbose) logTensor('shadows', shadows)

    const pass0 = new ShadowChebyshevDistancePass(shape, maxDistance)
    const t0 = runWebGLProgram(pass0, [shadows], 'int32', [], false)
    tf.dispose(shadows)

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

function extendedAnisotropicUnidirectionalDistanceMapInt32Array(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    tolerance: number, 
    blockSize: number,
    maxDistance: number,
    verbose: boolean = false,
): Int32Array
{
    const shadows = computeBidirectionalBlockShadowMap(volume, dominantAxis, octant, tolerance, blockSize, false)
    const shape = shadows.shape as [number, number, number]

    if (verbose) logTensor('shadows', shadows)

    const axes = ['x', 'y', 'z'] as [Axis, Axis, Axis]
    const dominantSign = signFromOctant(octant, axisIndex(dominantAxis))

    const otherAxes = axes.filter(a => a !== dominantAxis) as [Axis, Axis]
    const otherSigns = otherAxes.map(a => signFromOctant(octant, axisIndex(a))) as [Sign, Sign]

    const inAxes = [...otherAxes, dominantAxis] as [Axis, Axis, Axis]
    const inSigns = [...otherSigns, dominantSign] as [Sign, Sign, Sign]

    // initial distance
    const pass0 = new ShadowChebyshevDistancePass(shape, maxDistance)
    const t0 = runWebGLProgram(pass0, [shadows], 'int32', [], false)
    tf.dispose(shadows)

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

function extendedAnisotropicBidirectionalDistanceMapInt32Array(
    volume: tf.Tensor3D,
    dominantAxis: Axis,
    octant: Octant,
    tolerance: number, 
    blockSize: number,
    maxDistance: number,
    verbose: boolean = false,
): Int32Array
{
    const shadows = computeBidirectionalBlockShadowMap(volume, dominantAxis, octant, tolerance, blockSize, false)
    const shape = shadows.shape as [number, number, number]

    if (verbose) logTensor('shadows', shadows)

    const axes = ['x', 'y', 'z'] as [Axis, Axis, Axis]
    const dominantSign = signFromOctant(octant, axisIndex(dominantAxis))

    const otherAxes = axes.filter(a => a !== dominantAxis) as [Axis, Axis]
    const otherSigns = otherAxes.map(a => signFromOctant(octant, axisIndex(a))) as [Sign, Sign]

    const inAxes = [...otherAxes, dominantAxis] as [Axis, Axis, Axis]
    const inSigns = [...otherSigns, dominantSign] as [Sign, Sign, Sign]
    const invSigns = inSigns.map((s) => reverseSign(s))

    // initial distance
    const pass0 = new ShadowChebyshevDistancePass(shape, maxDistance)
    const t0 = runWebGLProgram(pass0, [shadows], 'int32', [], false)
    tf.dispose(shadows)

    // forward distance
    const pass1 = new AnisotropicChebyshevDistancePass(shape, inAxes[0], inSigns[0], maxDistance)
    const t1 = runWebGLProgram(pass1, [t0], 'int32', [], false)
    
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

type DistanceVariation = '1bit' | '5bit' | '8bit' | '10bit'
type DistanceMapSpec = [Axis, Octant, number]
type DistanceMapFn = (volume: tf.Tensor3D, dominantAxis: Axis, octant: Octant, tolerance: number, blockSize: number, maxDistance: number, verbose?: boolean) => Int32Array

function buildDistanceMapSpecs(maxDistanceXY: number, maxDistanceZ: number): Tuple<DistanceMapSpec, 12>
{
    return [
        ['x', '+++', maxDistanceXY],
        ['x', '+-+', maxDistanceXY],
        ['x', '++-', maxDistanceXY],
        ['x', '+--', maxDistanceXY],
        ['y', '+++', maxDistanceXY],
        ['y', '-++', maxDistanceXY],
        ['y', '++-', maxDistanceXY],
        ['y', '-+-', maxDistanceXY],
        ['z', '+++', maxDistanceZ],
        ['z', '+-+', maxDistanceZ],
        ['z', '-++', maxDistanceZ],
        ['z', '--+', maxDistanceZ],
    ] as Tuple<DistanceMapSpec, 12>
}

function createPackedDistanceBuffer(variation: DistanceVariation, voxels: number): Uint16Array | Uint32Array
{
    if (variation === '1bit') return new Uint16Array(voxels)
    if (variation === '5bit') return new Uint16Array(voxels * 4)
    if (variation === '8bit') return new Uint32Array(voxels * 3)

    return new Uint32Array(voxels * 4)
}

function packDistanceMapInPlace(packed: Uint16Array | Uint32Array, map: Int32Array, mapIndex: number, variation: DistanceVariation): void
{
    const voxels = map.length

    if (variation === '1bit')
    {
        const out = packed as Uint16Array
        const bit = mapIndex

        for (let i = 0; i < voxels; i++)
        {
            out[i] = out[i] | ((map[i] & 0x1) << bit)
        }

        return
    }

    if (variation === '5bit')
    {
        const out = packed as Uint16Array
        const channel = mapIndex & 3
        const axis = Math.floor(mapIndex / 4)
        const shift = (axis === 0) ? 0 : ((axis === 1) ? 5 : 10)
        const mask = (axis === 2) ? 0x3f : 0x1f

        for (let i = 0; i < voxels; i++)
        {
            const i4 = i * 4 + channel
            out[i4] = out[i4] | ((map[i] & mask) << shift)
        }

        return
    }

    if (variation === '8bit')
    {
        const out = packed as Uint32Array
        const axis = Math.floor(mapIndex / 4)
        const shift = (mapIndex & 3) * 8

        for (let i = 0; i < voxels; i++)
        {
            const i3 = i * 3 + axis
            out[i3] = (out[i3] | ((map[i] & 0xff) << shift)) >>> 0
        }

        return
    }

    const out = packed as Uint32Array
    const channel = mapIndex & 3
    const axis = Math.floor(mapIndex / 4)
    const shift = (axis === 0) ? 0 : ((axis === 1) ? 11 : 22)
    const mask = (axis === 2) ? 0x3ff : 0x7ff

    for (let i = 0; i < voxels; i++)
    {
        const i4 = i * 4 + channel
        out[i4] = (out[i4] | ((map[i] & mask) << shift)) >>> 0
    }
}

function computeDistanceMapPacked(
    volume: tf.Tensor3D,
    tolerance: number,
    blockSize: number,
    variation: DistanceVariation,
    maxDistanceXY: number,
    maxDistanceZ: number,
    mapFn: DistanceMapFn,
    verbose: boolean = false,
): Uint16Array | Uint32Array
{
    const specs = buildDistanceMapSpecs(maxDistanceXY, maxDistanceZ)

    let voxels = -1
    let packed: Uint16Array | Uint32Array | null = null

    for (let i = 0; i < 12; i++)
    {
        const [dominantAxis, octant, maxDistance] = specs[i]

        tf.engine().startScope()
        const map = mapFn(volume, dominantAxis, octant, tolerance, blockSize, maxDistance, verbose)
        tf.engine().endScope()

        if (voxels < 0)
        {
            voxels = map.length
            packed = createPackedDistanceBuffer(variation, voxels)
        }
        else if (map.length !== voxels)
        {
            throw new Error(`Expected all maps to have length ${voxels}, got ${map.length} at index ${i}`)
        }

        packDistanceMapInPlace(packed as Uint16Array | Uint32Array, map, i, variation)
    }

    return packed as Uint16Array | Uint32Array
}

export function computeIsotropicDistanceMap1bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint16Array
{
    return computeDistanceMapPacked(volume, tolerance, blockSize, '1bit', 1, 1, isotropicDistanceMapInt32Array, verbose) as Uint16Array
}

export function computeIsotropicDistanceMap5bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint16Array
{
    return computeDistanceMapPacked(volume, tolerance, blockSize, '5bit', 31, 63, isotropicDistanceMapInt32Array, verbose) as Uint16Array
}

export function computeIsotropicDistanceMap8bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint32Array
{
    return computeDistanceMapPacked(volume, tolerance, blockSize, '8bit', 255, 255, isotropicDistanceMapInt32Array, verbose) as Uint32Array
}

export function computeIsotropicDistanceMap10bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint32Array
{
    return computeDistanceMapPacked(volume, tolerance, blockSize, '10bit', 2047, 1023, isotropicDistanceMapInt32Array, verbose) as Uint32Array
}

export function computeExtendedAnisotropicUnidirectionalDistanceMap1bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint16Array
{
    return computeDistanceMapPacked(volume, tolerance, blockSize, '1bit', 1, 1, extendedAnisotropicUnidirectionalDistanceMapInt32Array, verbose) as Uint16Array
}

export function computeExtendedAnisotropicUnidirectionalDistanceMap5bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint16Array
{
    return computeDistanceMapPacked(volume, tolerance, blockSize, '5bit', 31, 63, extendedAnisotropicUnidirectionalDistanceMapInt32Array, verbose) as Uint16Array
}

export function computeExtendedAnisotropicUnidirectionalDistanceMap8bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint32Array
{
    return computeDistanceMapPacked(volume, tolerance, blockSize, '8bit', 255, 255, extendedAnisotropicUnidirectionalDistanceMapInt32Array, verbose) as Uint32Array
}

export function computeExtendedAnisotropicUnidirectionalDistanceMap10bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint32Array
{
    return computeDistanceMapPacked(volume, tolerance, blockSize, '10bit', 2047, 1023, extendedAnisotropicUnidirectionalDistanceMapInt32Array, verbose) as Uint32Array
}

export function computeExtendedAnisotropicBidirectionalDistanceMap1bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint16Array
{
    return computeDistanceMapPacked(volume, tolerance, blockSize, '1bit', 1, 1, extendedAnisotropicBidirectionalDistanceMapInt32Array, verbose) as Uint16Array
}

export function computeExtendedAnisotropicBidirectionalDistanceMap5bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint16Array
{
    return computeDistanceMapPacked(volume, tolerance, blockSize, '5bit', 31, 63, extendedAnisotropicBidirectionalDistanceMapInt32Array, verbose) as Uint16Array
}

export function computeExtendedAnisotropicBidirectionalDistanceMap8bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint32Array
{
    return computeDistanceMapPacked(volume, tolerance, blockSize, '8bit', 255, 255, extendedAnisotropicBidirectionalDistanceMapInt32Array, verbose) as Uint32Array
}

export function computeExtendedAnisotropicBidirectionalDistanceMap10bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint32Array
{
    return computeDistanceMapPacked(volume, tolerance, blockSize, '10bit', 2047, 1023, extendedAnisotropicBidirectionalDistanceMapInt32Array, verbose) as Uint32Array
}

