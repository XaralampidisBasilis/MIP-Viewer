import * as tf from '@tensorflow/tfjs'
import { GPGPUProgram } from '@tensorflow/tfjs-backend-webgl'
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl'
import { computeBidirectionalBlockShadowMap } from './GPGPUShadowMapPathsPacked'
import * as su from '../../Utils/ShadowMapUtils'

class InitialChebyshevDistancePass implements GPGPUProgram 
{
    variableNames = ['A'] 
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        shape: [number, number, number, 2, 2],
        maxDistance: number,
    ) {
        const [depth, height, width,] = shape

        maxDistance = Math.max(maxDistance, 0)
        maxDistance = Math.min(maxDistance, Math.max(depth, height, width))

        this.outputShape = shape
        this.userCode = `

        ivec3 outputCoords() 
        {
            ivec5 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        vec4 cellShadowsAt(ivec3 coords) 
        {
            return step(0.5, getA(coords.z, coords.y, coords.x, 0, 0));
        }

        void main() 
        {
            vec4 cellShadows = cellShadowsAt(outputCoords());
            vec4 cellDistances = mix(vec4(0), vec4(${maxDistance}), cellShadows);
            
            setOutput(cellDistances);
        }
        `
    }
}

class IsotropicChebyshevDistancePass implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        shape: [number, number, number, 2, 2],
        maxDistance: number,
        axis: su.Axis,
    ) { 
        const [depth, height, width,] = shape

        this.outputShape = shape
        this.userCode = /* glsl */ `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        const int maxRadius = clamp(${maxDistance}, minCoords.${axis}, maxCoords.${axis});

        bool insideAxis(ivec3 cellCoords)
        {
            return (cellCoords.${axis} >= minCoords.${axis} && cellCoords.${axis} <= maxCoords.${axis});
        }

        ivec3 outputCoords()
        {
            ivec5 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        ivec4 sampleChebyshevDistancesAt(ivec3 cellCoords)
        {
            return ivec4(getA(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0));
        }
    
        void main()
        {
            ivec3 outCoords = outputCoords();
            ivec3 sampleCoords = outCoords;

            ivec4 minDistances = sampleChebyshevDistancesAt(sampleCoords);

            if (all(lessThanEqual(minDistances, ivec4(1)))
            {
                setOutput(vec4(minDistances));
                return;
            }

            for (int radius = 1; radius <= maxRadius; ++radius)
            {
                sampleCoords.${axis} = outCoords.${axis} - radius;
                
                if (insideAxis(sampleCoords))
                {
                    ivec4 sampleDistances = sampleChebyshevDistancesAt(sampleCoords);

                    ivec4 candidateDistances = max(sampleDistances, radius);
                    minDistances = min(minDistances, candidateDistances);

                    if (all(lessThanEqual(minDistances, ivec4(radius))) 
                    {
                        break;
                    }
                }

                sampleCoords.${axis} = outCoords.${axis} + radius;

                if (insideAxis(sampleCoords))
                {
                    ivec4 sampleDistances = sampleChebyshevDistancesAt(sampleCoords);

                    ivec4 candidateDistances = max(sampleDistances, radius);
                    minDistances = min(minDistances, candidateDistances);

                    if (all(lessThanEqual(minDistances, ivec4(radius))) 
                    {
                        break;
                    }
                }
            }

            setOutput(vec4(minDistances));
        }
        `
    }
}

class AnisotropicChebyshevDistancePass implements GPGPUProgram
{
    variableNames = ['A']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        shape: [number, number, number, 2, 2],
        maxDistance: number,
        axis: su.Axis,
        sign: su.Sign,
    ) {
        const [depth, height, width,] = shape

        this.outputShape = shape
        this.userCode = /* glsl */ `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});
        const int maxRadius = clamp(${maxDistance}, minCoords.${axis}, maxCoords.${axis});

        bool insideAxis(ivec3 cellCoords)
        {
            return (cellCoords.${axis} >= minCoords.${axis} && cellCoords.${axis} <= maxCoords.${axis});
        }

        ivec3 outputCoords()
        {
            ivec5 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        ivec4 sampleChebyshevDistancesAt(ivec3 cellCoords)
        {
            return ivec4(getA(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0));
        }

        void main()
        {
            ivec3 outCoords = outputCoords();
            ivec3 sampleCoords = outCoords;

            ivec4 minDistances = sampleChebyshevDistancesAt(sampleCoords);

            if (all(lessThanEqual(minDistances, ivec4(1)))
            {
                setOutput(vec4(minDistances));
                return;
            }

            for (int radius = 1; radius <= maxRadius; ++radius)
            {
                sampleCoords.${axis} = outCoords.${axis} ${sign} radius;

                if (!insideAxis(sampleCoords)) 
                {
                    break;
                }

                ivec4 sampleDistances = sampleChebyshevDistancesAt(sampleCoords);

                ivec4 candidateDistances = max(sampleDistances, radius);
                minDistances = min(minDistances, candidateDistances);

                if (all(lessThanEqual(minDistances, ivec4(radius)))
                {
                    break;
                }
                
            }

            setOutput(vec4(minDistances));
        }
        `
    }
}

class ExtendedChebyshevDistancePass implements GPGPUProgram 
{
    variableNames = ['A']
    outputShape: [number, number, number, 2, 2]
    userCode: string
    packedInputs = true
    packedOutput = true

    constructor(
        shape: [number, number, number, 2, 2],
        axis: su.Axis,
        sign: su.Sign,        
        maxDistance: number,
    ) {
        const [depth, height, width,] = shape

        this.outputShape = shape
        this.userCode = /* glsl */ `
        const ivec3 minCoords = ivec3(0);
        const ivec3 maxCoords = ivec3(${width - 1}, ${height - 1}, ${depth - 1});

        const int maxRadius = clamp(${maxDistance}, minCoords.${axis}, maxCoords.${axis});

        bool insideAxis(ivec3 cellCoords)
        {
            return (cellCoords.${axis} >= minCoords.${axis} && cellCoords.${axis} <= maxCoords.${axis});
        }

        ivec3 outputCoords()
        {
            ivec5 cellCoords = getOutputCoords();
            return ivec3(cellCoords.z, cellCoords.y, cellCoords.x);
        }

        ivec4 sampleChebyshevDistancesAt(ivec3 cellCoords)
        {
            return ivec4(getA(cellCoords.z, cellCoords.y, cellCoords.x, 0, 0));
        }

        void main() 
        {
            ivec3 outCoords = outputCoords();
            ivec3 sampleCoords = outCoords;

            int minDistance = sampleChebyshevDistancesAt(sampleCoords);
            if (minDistance == 0) 
            {
                setOutput(minDistance);
                return;
            }

            minDistance = ${maxDistance};

            for (int radius = 1; radius <= maxRadius; ++radius) 
            {
                sampleCoords.${axis} = outCoords.${axis} ${sign} radius;

                if (!insideAxis(sampleCoords)) 
                {
                    break;
                }

                int sampleDistance = sampleChebyshevDistancesAt(sampleCoords);

                if (sampleDistance <= radius) 
                {
                    minDistance = radius;
                    break;
                }
            }

            setOutput(float(minDistance));
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

function packToR16UIArray(maps: su.Tuple<Int32Array, 12>): Uint16Array 
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

function packToRGBA16UIArray(maps: su.Tuple<Int32Array, 12>): Uint16Array 
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

function packToRGB32UIArray(maps: su.Tuple<Int32Array, 12>): Uint32Array 
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

function packToRGBA32UIArray(maps: su.Tuple<Int32Array, 12>): Uint32Array 
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
    dominantAxis: su.Axis,
    octant: su.Octant,
    tolerance: number, 
    blockSize: number,
    maxDistance: number,
    verbose: boolean = false,
): Int32Array
{
    const shadows = computeBidirectionalBlockShadowMap(volume, dominantAxis, octant, tolerance, blockSize, false)
    const shape = shadows.shape as [number, number, number]

    if (verbose) logTensor('shadows', shadows)
    
    const pass0 = new InitialChebyshevDistancePass(shape, maxDistance)
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
    dominantAxis: su.Axis,
    octant: su.Octant,
    tolerance: number, 
    blockSize: number,
    maxDistance: number,
    verbose: boolean = false,
): Int32Array
{
    const shadows = computeBidirectionalBlockShadowMap(volume, dominantAxis, octant, tolerance, blockSize, false)
    const shape = shadows.shape as [number, number, number]

    if (verbose) logTensor('shadows', shadows)

    const axes = ['x', 'y', 'z'] as [su.Axis, su.Axis, su.Axis]
    const dominantSign = su.getOctantSign(octant, su.axisToDimension(dominantAxis))

    const otherAxes = axes.filter(a => a !== dominantAxis) as [su.Axis, su.Axis]
    const otherSigns = otherAxes.map(a => su.getOctantSign(octant, su.axisToDimension(a))) as [su.Sign, su.Sign]

    const inAxes = [...otherAxes, dominantAxis] as [su.Axis, su.Axis, su.Axis]
    const inSigns = [...otherSigns, dominantSign] as [su.Sign, su.Sign, su.Sign]

    const pass0 = new InitialChebyshevDistancePass(shape, maxDistance)
    const t0 = runWebGLProgram(pass0, [shadows], 'int32', [], false)
    tf.dispose(shadows)
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
    dominantAxis: su.Axis,
    octant: su.Octant,
    tolerance: number, 
    blockSize: number,
    maxDistance: number,
    verbose: boolean = false,
): Int32Array
{
    const shadows = computeBidirectionalBlockShadowMap(volume, dominantAxis, octant, tolerance, blockSize, false)
    const shape = shadows.shape as [number, number, number]

    if (verbose) logTensor('shadows', shadows)

    const axes = ['x', 'y', 'z'] as [su.Axis, su.Axis, su.Axis]
    const dominantSign = su.getOctantSign(octant, su.axisToDimension(dominantAxis))

    const otherAxes = axes.filter(a => a !== dominantAxis) as [su.Axis, su.Axis]
    const otherSigns = otherAxes.map(a => su.getOctantSign(octant, su.axisToDimension(a))) as [su.Sign, su.Sign]

    const inAxes = [...otherAxes, dominantAxis] as [su.Axis, su.Axis, su.Axis]
    const inSigns = [...otherSigns, dominantSign] as [su.Sign, su.Sign, su.Sign]
    const invSigns = inSigns.map((s) => su.reverseSign(s))


    // initial distance
    const pass0 = new InitialChebyshevDistancePass(shape, maxDistance)
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


export function computeIsotropicDistanceMap1bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint16Array
{
    const maps = new Array(12) 

    maps[ 0] = isotropicDistanceMapInt32Array(volume, 'x', '+++', tolerance, blockSize, 1, verbose)
    maps[ 1] = isotropicDistanceMapInt32Array(volume, 'x', '+-+', tolerance, blockSize, 1, verbose)
    maps[ 2] = isotropicDistanceMapInt32Array(volume, 'x', '++-', tolerance, blockSize, 1, verbose)
    maps[ 3] = isotropicDistanceMapInt32Array(volume, 'x', '+--', tolerance, blockSize, 1, verbose)
    maps[ 4] = isotropicDistanceMapInt32Array(volume, 'y', '+++', tolerance, blockSize, 1, verbose)
    maps[ 5] = isotropicDistanceMapInt32Array(volume, 'y', '-++', tolerance, blockSize, 1, verbose)
    maps[ 6] = isotropicDistanceMapInt32Array(volume, 'y', '++-', tolerance, blockSize, 1, verbose)
    maps[ 7] = isotropicDistanceMapInt32Array(volume, 'y', '-+-', tolerance, blockSize, 1, verbose)
    maps[ 8] = isotropicDistanceMapInt32Array(volume, 'z', '+++', tolerance, blockSize, 1, verbose)
    maps[ 9] = isotropicDistanceMapInt32Array(volume, 'z', '+-+', tolerance, blockSize, 1, verbose)
    maps[10] = isotropicDistanceMapInt32Array(volume, 'z', '-++', tolerance, blockSize, 1, verbose)
    maps[11] = isotropicDistanceMapInt32Array(volume, 'z', '--+', tolerance, blockSize, 1, verbose)

    return packToR16UIArray(maps as su.Tuple<Int32Array, 12>) 
}

export function computeIsotropicDistanceMap5bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint16Array
{
    const maps = new Array(12) 

    maps[ 0] = isotropicDistanceMapInt32Array(volume, 'x', '+++', tolerance, blockSize, 31, verbose)
    maps[ 1] = isotropicDistanceMapInt32Array(volume, 'x', '+-+', tolerance, blockSize, 31, verbose)
    maps[ 2] = isotropicDistanceMapInt32Array(volume, 'x', '++-', tolerance, blockSize, 31, verbose)
    maps[ 3] = isotropicDistanceMapInt32Array(volume, 'x', '+--', tolerance, blockSize, 31, verbose)
    maps[ 4] = isotropicDistanceMapInt32Array(volume, 'y', '+++', tolerance, blockSize, 31, verbose)
    maps[ 5] = isotropicDistanceMapInt32Array(volume, 'y', '-++', tolerance, blockSize, 31, verbose)
    maps[ 6] = isotropicDistanceMapInt32Array(volume, 'y', '++-', tolerance, blockSize, 31, verbose)
    maps[ 7] = isotropicDistanceMapInt32Array(volume, 'y', '-+-', tolerance, blockSize, 31, verbose)
    maps[ 8] = isotropicDistanceMapInt32Array(volume, 'z', '+++', tolerance, blockSize, 63, verbose)
    maps[ 9] = isotropicDistanceMapInt32Array(volume, 'z', '+-+', tolerance, blockSize, 63, verbose)
    maps[10] = isotropicDistanceMapInt32Array(volume, 'z', '-++', tolerance, blockSize, 63, verbose)
    maps[11] = isotropicDistanceMapInt32Array(volume, 'z', '--+', tolerance, blockSize, 63, verbose)

    return packToRGBA16UIArray(maps as su.Tuple<Int32Array, 12>) 
}

export function computeIsotropicDistanceMap8bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint32Array
{
    const maps = new Array(12)

    maps[ 0] = isotropicDistanceMapInt32Array(volume, 'x', '+++', tolerance, blockSize, 255, verbose)
    maps[ 1] = isotropicDistanceMapInt32Array(volume, 'x', '+-+', tolerance, blockSize, 255, verbose)
    maps[ 2] = isotropicDistanceMapInt32Array(volume, 'x', '++-', tolerance, blockSize, 255, verbose)
    maps[ 3] = isotropicDistanceMapInt32Array(volume, 'x', '+--', tolerance, blockSize, 255, verbose)
    maps[ 4] = isotropicDistanceMapInt32Array(volume, 'y', '+++', tolerance, blockSize, 255, verbose)
    maps[ 5] = isotropicDistanceMapInt32Array(volume, 'y', '-++', tolerance, blockSize, 255, verbose)
    maps[ 6] = isotropicDistanceMapInt32Array(volume, 'y', '++-', tolerance, blockSize, 255, verbose)
    maps[ 7] = isotropicDistanceMapInt32Array(volume, 'y', '-+-', tolerance, blockSize, 255, verbose)
    maps[ 8] = isotropicDistanceMapInt32Array(volume, 'z', '+++', tolerance, blockSize, 255, verbose)
    maps[ 9] = isotropicDistanceMapInt32Array(volume, 'z', '+-+', tolerance, blockSize, 255, verbose)
    maps[10] = isotropicDistanceMapInt32Array(volume, 'z', '-++', tolerance, blockSize, 255, verbose)
    maps[11] = isotropicDistanceMapInt32Array(volume, 'z', '--+', tolerance, blockSize, 255, verbose)

    return packToRGB32UIArray(maps as su.Tuple<Int32Array, 12>) 
}

export function computeIsotropicDistanceMap10bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint32Array
{
    const maps = new Array(12)

    maps[ 0] = isotropicDistanceMapInt32Array(volume, 'x', '+++', tolerance, blockSize, 2047, verbose)
    maps[ 1] = isotropicDistanceMapInt32Array(volume, 'x', '+-+', tolerance, blockSize, 2047, verbose)
    maps[ 2] = isotropicDistanceMapInt32Array(volume, 'x', '++-', tolerance, blockSize, 2047, verbose)
    maps[ 3] = isotropicDistanceMapInt32Array(volume, 'x', '+--', tolerance, blockSize, 2047, verbose)
    maps[ 4] = isotropicDistanceMapInt32Array(volume, 'y', '+++', tolerance, blockSize, 2047, verbose)
    maps[ 5] = isotropicDistanceMapInt32Array(volume, 'y', '-++', tolerance, blockSize, 2047, verbose)
    maps[ 6] = isotropicDistanceMapInt32Array(volume, 'y', '++-', tolerance, blockSize, 2047, verbose)
    maps[ 7] = isotropicDistanceMapInt32Array(volume, 'y', '-+-', tolerance, blockSize, 2047, verbose)
    maps[ 8] = isotropicDistanceMapInt32Array(volume, 'z', '+++', tolerance, blockSize, 1023, verbose)
    maps[ 9] = isotropicDistanceMapInt32Array(volume, 'z', '+-+', tolerance, blockSize, 1023, verbose)
    maps[10] = isotropicDistanceMapInt32Array(volume, 'z', '-++', tolerance, blockSize, 1023, verbose)
    maps[11] = isotropicDistanceMapInt32Array(volume, 'z', '--+', tolerance, blockSize, 1023, verbose)

    return packToRGBA32UIArray(maps as su.Tuple<Int32Array, 12>) 
}


export function computeExtendedAnisotropicUnidirectionalDistanceMap1bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint16Array
{
    const maps = new Array(12) 

    maps[ 0] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '+++', tolerance, blockSize, 1, verbose)
    maps[ 1] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '+-+', tolerance, blockSize, 1, verbose)
    maps[ 2] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '++-', tolerance, blockSize, 1, verbose)
    maps[ 3] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '+--', tolerance, blockSize, 1, verbose)
    maps[ 4] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '+++', tolerance, blockSize, 1, verbose)
    maps[ 5] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '-++', tolerance, blockSize, 1, verbose)
    maps[ 6] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '++-', tolerance, blockSize, 1, verbose)
    maps[ 7] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '-+-', tolerance, blockSize, 1, verbose)
    maps[ 8] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '+++', tolerance, blockSize, 1, verbose)
    maps[ 9] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '+-+', tolerance, blockSize, 1, verbose)
    maps[10] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '-++', tolerance, blockSize, 1, verbose)
    maps[11] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '--+', tolerance, blockSize, 1, verbose)

    return packToR16UIArray(maps as su.Tuple<Int32Array, 12>) 
}

export function computeExtendedAnisotropicUnidirectionalDistanceMap5bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint16Array
{
    const maps = new Array(12) 

    maps[ 0] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '+++', tolerance, blockSize, 31, verbose)
    maps[ 1] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '+-+', tolerance, blockSize, 31, verbose)
    maps[ 2] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '++-', tolerance, blockSize, 31, verbose)
    maps[ 3] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '+--', tolerance, blockSize, 31, verbose)
    maps[ 4] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '+++', tolerance, blockSize, 31, verbose)
    maps[ 5] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '-++', tolerance, blockSize, 31, verbose)
    maps[ 6] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '++-', tolerance, blockSize, 31, verbose)
    maps[ 7] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '-+-', tolerance, blockSize, 31, verbose)
    maps[ 8] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '+++', tolerance, blockSize, 63, verbose)
    maps[ 9] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '+-+', tolerance, blockSize, 63, verbose)
    maps[10] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '-++', tolerance, blockSize, 63, verbose)
    maps[11] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '--+', tolerance, blockSize, 63, verbose)

    return packToRGBA16UIArray(maps as su.Tuple<Int32Array, 12>) 
}

export function computeExtendedAnisotropicUnidirectionalDistanceMap8bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint32Array
{
    const maps = new Array(12)

    maps[ 0] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '+++', tolerance, blockSize, 255, verbose)
    maps[ 1] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '+-+', tolerance, blockSize, 255, verbose)
    maps[ 2] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '++-', tolerance, blockSize, 255, verbose)
    maps[ 3] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '+--', tolerance, blockSize, 255, verbose)
    maps[ 4] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '+++', tolerance, blockSize, 255, verbose)
    maps[ 5] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '-++', tolerance, blockSize, 255, verbose)
    maps[ 6] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '++-', tolerance, blockSize, 255, verbose)
    maps[ 7] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '-+-', tolerance, blockSize, 255, verbose)
    maps[ 8] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '+++', tolerance, blockSize, 255, verbose)
    maps[ 9] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '+-+', tolerance, blockSize, 255, verbose)
    maps[10] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '-++', tolerance, blockSize, 255, verbose)
    maps[11] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '--+', tolerance, blockSize, 255, verbose)

    return packToRGB32UIArray(maps as su.Tuple<Int32Array, 12>) 
}

export function computeExtendedAnisotropicUnidirectionalDistanceMap10bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint32Array
{
    const maps = new Array(12)

    maps[ 0] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '+++', tolerance, blockSize, 2047, verbose)
    maps[ 1] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '+-+', tolerance, blockSize, 2047, verbose)
    maps[ 2] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '++-', tolerance, blockSize, 2047, verbose)
    maps[ 3] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'x', '+--', tolerance, blockSize, 2047, verbose)
    maps[ 4] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '+++', tolerance, blockSize, 2047, verbose)
    maps[ 5] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '-++', tolerance, blockSize, 2047, verbose)
    maps[ 6] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '++-', tolerance, blockSize, 2047, verbose)
    maps[ 7] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'y', '-+-', tolerance, blockSize, 2047, verbose)
    maps[ 8] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '+++', tolerance, blockSize, 1023, verbose)
    maps[ 9] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '+-+', tolerance, blockSize, 1023, verbose)
    maps[10] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '-++', tolerance, blockSize, 1023, verbose)
    maps[11] = extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '--+', tolerance, blockSize, 1023, verbose)

    return packToRGBA32UIArray(maps as su.Tuple<Int32Array, 12>) 
}


export function computeExtendedAnisotropicBidirectionalDistanceMap1bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint16Array
{
    const maps = new Array(12) 

    maps[ 0] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '+++', tolerance, blockSize, 1, verbose)
    maps[ 1] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '+-+', tolerance, blockSize, 1, verbose)
    maps[ 2] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '++-', tolerance, blockSize, 1, verbose)
    maps[ 3] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '+--', tolerance, blockSize, 1, verbose)
    maps[ 4] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '+++', tolerance, blockSize, 1, verbose)
    maps[ 5] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '-++', tolerance, blockSize, 1, verbose)
    maps[ 6] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '++-', tolerance, blockSize, 1, verbose)
    maps[ 7] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '-+-', tolerance, blockSize, 1, verbose)
    maps[ 8] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '+++', tolerance, blockSize, 1, verbose)
    maps[ 9] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '+-+', tolerance, blockSize, 1, verbose)
    maps[10] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '-++', tolerance, blockSize, 1, verbose)
    maps[11] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '--+', tolerance, blockSize, 1, verbose)

    return packToR16UIArray(maps as su.Tuple<Int32Array, 12>) 
}

export function computeExtendedAnisotropicBidirectionalDistanceMap5bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint16Array
{
    const maps = new Array(12) 

    maps[ 0] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '+++', tolerance, blockSize, 31, verbose)
    maps[ 1] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '+-+', tolerance, blockSize, 31, verbose)
    maps[ 2] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '++-', tolerance, blockSize, 31, verbose)
    maps[ 3] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '+--', tolerance, blockSize, 31, verbose)
    maps[ 4] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '+++', tolerance, blockSize, 31, verbose)
    maps[ 5] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '-++', tolerance, blockSize, 31, verbose)
    maps[ 6] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '++-', tolerance, blockSize, 31, verbose)
    maps[ 7] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '-+-', tolerance, blockSize, 31, verbose)
    maps[ 8] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '+++', tolerance, blockSize, 63, verbose)
    maps[ 9] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '+-+', tolerance, blockSize, 63, verbose)
    maps[10] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '-++', tolerance, blockSize, 63, verbose)
    maps[11] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '--+', tolerance, blockSize, 63, verbose)

    return packToRGBA16UIArray(maps as su.Tuple<Int32Array, 12>) 
}

export function computeExtendedAnisotropicBidirectionalDistanceMap8bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint32Array
{
    const maps = new Array(12)

    maps[ 0] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '+++', tolerance, blockSize, 255, verbose)
    maps[ 1] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '+-+', tolerance, blockSize, 255, verbose)
    maps[ 2] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '++-', tolerance, blockSize, 255, verbose)
    maps[ 3] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '+--', tolerance, blockSize, 255, verbose)
    maps[ 4] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '+++', tolerance, blockSize, 255, verbose)
    maps[ 5] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '-++', tolerance, blockSize, 255, verbose)
    maps[ 6] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '++-', tolerance, blockSize, 255, verbose)
    maps[ 7] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '-+-', tolerance, blockSize, 255, verbose)
    maps[ 8] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '+++', tolerance, blockSize, 255, verbose)
    maps[ 9] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '+-+', tolerance, blockSize, 255, verbose)
    maps[10] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '-++', tolerance, blockSize, 255, verbose)
    maps[11] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '--+', tolerance, blockSize, 255, verbose)

    return packToRGB32UIArray(maps as su.Tuple<Int32Array, 12>) 
}

export function computeExtendedAnisotropicBidirectionalDistanceMap10bit(volume: tf.Tensor3D, tolerance: number, blockSize: number, verbose: boolean = false) : Uint32Array
{
    const maps = new Array(12)

    maps[ 0] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '+++', tolerance, blockSize, 2047, verbose)
    maps[ 1] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '+-+', tolerance, blockSize, 2047, verbose)
    maps[ 2] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '++-', tolerance, blockSize, 2047, verbose)
    maps[ 3] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'x', '+--', tolerance, blockSize, 2047, verbose)
    maps[ 4] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '+++', tolerance, blockSize, 2047, verbose)
    maps[ 5] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '-++', tolerance, blockSize, 2047, verbose)
    maps[ 6] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '++-', tolerance, blockSize, 2047, verbose)
    maps[ 7] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'y', '-+-', tolerance, blockSize, 2047, verbose)
    maps[ 8] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '+++', tolerance, blockSize, 1023, verbose)
    maps[ 9] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '+-+', tolerance, blockSize, 1023, verbose)
    maps[10] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '-++', tolerance, blockSize, 1023, verbose)
    maps[11] = extendedAnisotropicBidirectionalDistanceMapInt32Array(volume, 'z', '--+', tolerance, blockSize, 1023, verbose)

    return packToRGBA32UIArray(maps as su.Tuple<Int32Array, 12>) 
}

