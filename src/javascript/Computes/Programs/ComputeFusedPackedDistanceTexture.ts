import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import * as FusedDistanceMaps from './GPGPUDistanceMapFused'
import * as FusedShadowMaps from './GPGPUShadowMapPathsFused'
import { type Axis, type Sign, type Tuple } from '../../Utils/ShadowMapUtils'

export type DistanceVariant = 'isotropic' | 'unidirectional' | 'bidirectional'
export type DistanceEncoding = '1bit' | '5bit' | '8bit' | '10bit'

export type FusedDistanceArrays = Tuple<Int32Array, 3>
export type PackedDistanceMaps = Uint16Array | Uint32Array

export type PackedDistanceTexture =
{
    data: PackedDistanceMaps
    texture: THREE.Data3DTexture
    dimensions: THREE.Vector3
    encoding: DistanceEncoding
}

type MaxDistances = Record<Axis, number>
type IntegerTextureFormat = 'RED_INTEGER' | 'RGB_INTEGER' | 'RGBA_INTEGER'

type DistanceTextureFormat =
{
    format: IntegerTextureFormat
    type: number
    internalFormat: string
}

const FUSED_DISTANCE_AXES: Tuple<Axis, 3> =
[
    'x',
    'y',
    'z',
]

const MAX_ENCODABLE_DISTANCES: Record<DistanceEncoding, MaxDistances> =
{
    '1bit' : { x: 1,    y: 1,    z: 1    },
    '5bit' : { x: 31,   y: 31,   z: 63   },
    '8bit' : { x: 255,  y: 255,  z: 255  },
    '10bit': { x: 2047, y: 2047, z: 1023 },
}

const FUSED_DISTANCE_PACKERS: Record<DistanceEncoding, (arrays: FusedDistanceArrays) => PackedDistanceMaps> =
{
    '1bit' : packFused1Bit,
    '5bit' : packFused5Bit,
    '8bit' : packFused8Bit,
    '10bit': packFused10Bit,
}

// This THREE version may not map THREE.RGBIntegerFormat to WebGL's RGB_INTEGER
// for Data3DTexture uploads, so use the raw WebGL enum name string instead.
const DISTANCE_TEXTURE_FORMATS: Record<DistanceEncoding, DistanceTextureFormat> =
{
    '1bit' : { format: 'RED_INTEGER',  type: THREE.UnsignedShortType, internalFormat: 'R16UI'    },
    '5bit' : { format: 'RGBA_INTEGER', type: THREE.UnsignedShortType, internalFormat: 'RGBA16UI' },
    '8bit' : { format: 'RGB_INTEGER',  type: THREE.UnsignedIntType,   internalFormat: 'RGB32UI'  },
    '10bit': { format: 'RGBA_INTEGER', type: THREE.UnsignedIntType,   internalFormat: 'RGBA32UI' },
}

function maxEncodableDistance(encoding: DistanceEncoding, axis: Axis): number
{
    return MAX_ENCODABLE_DISTANCES[encoding][axis]
}

function computeTextureDimensions(volume: tf.Tensor3D, blockSize: number): THREE.Vector3
{
    const blockShape = volume.shape.map((x) => Math.ceil((x + 1) / blockSize))
    const [depth, height, width] = blockShape

    return new THREE.Vector3(width, height, depth)
}

function voxelCountFromFusedArray(array: Int32Array, name: string): number
{
    if (array.length % 4 !== 0)
    {
        throw new Error(`${name} expected fused array length to be divisible by 4, got ${array.length}`)
    }

    return array.length / 4
}

function assertSameFusedVoxelCount(arrays: FusedDistanceArrays, name: string): number
{
    const voxels = voxelCountFromFusedArray(arrays[0], name)

    for (let i = 1; i < arrays.length; i += 1)
    {
        const currentVoxels = voxelCountFromFusedArray(arrays[i], name)

        if (currentVoxels !== voxels)
        {
            throw new Error(`${name} expected fused array ${i} to contain ${voxels} voxels, got ${currentVoxels}`)
        }
    }

    return voxels
}

function packFusedDistanceArrays(
    arrays: FusedDistanceArrays,
    encoding: DistanceEncoding,
): PackedDistanceMaps
{
    return FUSED_DISTANCE_PACKERS[encoding](arrays)
}

function packFused1Bit(arrays: FusedDistanceArrays): Uint16Array
{
    const voxels = assertSameFusedVoxelCount(arrays, 'packFused1Bit')
    const packed = new Uint16Array(voxels)

    const x = arrays[0]
    const y = arrays[1]
    const z = arrays[2]

    for (let i = 0; i < voxels; i += 1)
    {
        const i4 = i * 4

        const x0 = x[i4 + 0] & 0x1
        const x1 = x[i4 + 1] & 0x1
        const x2 = x[i4 + 2] & 0x1
        const x3 = x[i4 + 3] & 0x1

        const y0 = y[i4 + 0] & 0x1
        const y1 = y[i4 + 1] & 0x1
        const y2 = y[i4 + 2] & 0x1
        const y3 = y[i4 + 3] & 0x1

        const z0 = z[i4 + 0] & 0x1
        const z1 = z[i4 + 1] & 0x1
        const z2 = z[i4 + 2] & 0x1
        const z3 = z[i4 + 3] & 0x1

        packed[i] =
        (x0 << 0) | (x1 << 1) | (x2 <<  2) | (x3 <<  3) |
        (y0 << 4) | (y1 << 5) | (y2 <<  6) | (y3 <<  7) |
        (z0 << 8) | (z1 << 9) | (z2 << 10) | (z3 << 11)
    }

    return packed
}

function packFused5Bit(arrays: FusedDistanceArrays): Uint16Array
{
    const voxels = assertSameFusedVoxelCount(arrays, 'packFused5Bit')
    const packed = new Uint16Array(voxels * 4)

    const x = arrays[0]
    const y = arrays[1]
    const z = arrays[2]

    for (let i = 0; i < voxels; i += 1)
    {
        const i4 = i * 4

        const x0 = x[i4 + 0] & 0x1f
        const x1 = x[i4 + 1] & 0x1f
        const x2 = x[i4 + 2] & 0x1f
        const x3 = x[i4 + 3] & 0x1f

        const y0 = y[i4 + 0] & 0x1f
        const y1 = y[i4 + 1] & 0x1f
        const y2 = y[i4 + 2] & 0x1f
        const y3 = y[i4 + 3] & 0x1f

        const z0 = z[i4 + 0] & 0x3f
        const z1 = z[i4 + 1] & 0x3f
        const z2 = z[i4 + 2] & 0x3f
        const z3 = z[i4 + 3] & 0x3f

        packed[i4 + 0] = (x0 | (y0 << 5) | (z0 << 10)) >>> 0
        packed[i4 + 1] = (x1 | (y1 << 5) | (z1 << 10)) >>> 0
        packed[i4 + 2] = (x2 | (y2 << 5) | (z2 << 10)) >>> 0
        packed[i4 + 3] = (x3 | (y3 << 5) | (z3 << 10)) >>> 0
    }

    return packed
}

function packFused8Bit(arrays: FusedDistanceArrays): Uint32Array
{
    const voxels = assertSameFusedVoxelCount(arrays, 'packFused8Bit')
    const packed = new Uint32Array(voxels * 3)

    const x = arrays[0]
    const y = arrays[1]
    const z = arrays[2]

    for (let i = 0; i < voxels; i += 1)
    {
        const source = i * 4
        const target = i * 3

        const x0 = x[source + 0] & 0xff
        const x1 = x[source + 1] & 0xff
        const x2 = x[source + 2] & 0xff
        const x3 = x[source + 3] & 0xff

        const y0 = y[source + 0] & 0xff
        const y1 = y[source + 1] & 0xff
        const y2 = y[source + 2] & 0xff
        const y3 = y[source + 3] & 0xff

        const z0 = z[source + 0] & 0xff
        const z1 = z[source + 1] & 0xff
        const z2 = z[source + 2] & 0xff
        const z3 = z[source + 3] & 0xff

        packed[target + 0] = (x0 | (x1 << 8) | (x2 << 16) | (x3 << 24)) >>> 0
        packed[target + 1] = (y0 | (y1 << 8) | (y2 << 16) | (y3 << 24)) >>> 0
        packed[target + 2] = (z0 | (z1 << 8) | (z2 << 16) | (z3 << 24)) >>> 0
    }

    return packed
}

function packFused10Bit(arrays: FusedDistanceArrays): Uint32Array
{
    const voxels = assertSameFusedVoxelCount(arrays, 'packFused10Bit')
    const packed = new Uint32Array(voxels * 4)

    const x = arrays[0]
    const y = arrays[1]
    const z = arrays[2]

    for (let i = 0; i < voxels; i += 1)
    {
        const i4 = i * 4

        const x0 = x[i4 + 0] & 0x7ff
        const x1 = x[i4 + 1] & 0x7ff
        const x2 = x[i4 + 2] & 0x7ff
        const x3 = x[i4 + 3] & 0x7ff

        const y0 = y[i4 + 0] & 0x7ff
        const y1 = y[i4 + 1] & 0x7ff
        const y2 = y[i4 + 2] & 0x7ff
        const y3 = y[i4 + 3] & 0x7ff

        const z0 = z[i4 + 0] & 0x3ff
        const z1 = z[i4 + 1] & 0x3ff
        const z2 = z[i4 + 2] & 0x3ff
        const z3 = z[i4 + 3] & 0x3ff

        packed[i4 + 0] = (x0 | (y0 << 11) | (z0 << 22)) >>> 0
        packed[i4 + 1] = (x1 | (y1 << 11) | (z1 << 22)) >>> 0
        packed[i4 + 2] = (x2 | (y2 << 11) | (z2 << 22)) >>> 0
        packed[i4 + 3] = (x3 | (y3 << 11) | (z3 << 22)) >>> 0
    }

    return packed
}

function computeFusedDistanceTensor(
    volume: tf.Tensor3D,
    variant: DistanceVariant,
    axis: Axis,
    sign: Sign,
    tolerance: number,
    blockSize: number,
    maxDistance: number,
    verbose: boolean,
): tf.Tensor5D
{
    const shadowMaps = FusedShadowMaps.computeBidirectionalShadowMaps(volume, axis, sign, tolerance, blockSize)

    let distanceMaps: tf.Tensor5D | undefined

    try
    {
        switch (variant)
        {
            case 'isotropic':
                distanceMaps = FusedDistanceMaps.computeIsotropicDistanceMaps(shadowMaps, maxDistance, verbose)
                break

            case 'unidirectional':
                distanceMaps = FusedDistanceMaps.computeUnidirectionalDistanceMaps(shadowMaps, axis, sign, maxDistance, verbose)
                break

            case 'bidirectional':
                distanceMaps = FusedDistanceMaps.computeBidirectionalDistanceMaps(shadowMaps, axis, sign, maxDistance, verbose)
                break
        }

        return distanceMaps
    }
    finally
    {
        shadowMaps.dispose()
    }
}

export function computeFusedDistanceArrays(
    volume: tf.Tensor3D,
    variant: DistanceVariant,
    tolerance: number,
    blockSize: number,
    encoding: DistanceEncoding,
    verbose: boolean = false,
): FusedDistanceArrays
{
    const arrays = FUSED_DISTANCE_AXES.map((axis) =>
    {
        const maxDistance = maxEncodableDistance(encoding, axis)

        const distanceMaps = computeFusedDistanceTensor( volume, variant, axis, '+', tolerance, blockSize, maxDistance, verbose)

        try
        {
            return distanceMaps.dataSync() as Int32Array
        }
        finally
        {
            distanceMaps.dispose()
        }
    })

    return arrays as FusedDistanceArrays
}

export function computePackedDistanceMaps(
    volume: tf.Tensor3D,
    variant: DistanceVariant,
    tolerance: number,
    blockSize: number,
    encoding: DistanceEncoding,
    verbose: boolean = false,
): PackedDistanceMaps
{
    const arrays = computeFusedDistanceArrays( volume, variant, tolerance, blockSize, encoding, verbose)

    return packFusedDistanceArrays(arrays, encoding)
}

export function computePackedDistanceTexture(
    volume: tf.Tensor3D,
    variant: DistanceVariant,
    tolerance: number,
    blockSize: number,
    encoding: DistanceEncoding,
    verbose: boolean = false,
): PackedDistanceTexture
{
    const textureFormat = DISTANCE_TEXTURE_FORMATS[encoding]
    const dimensions = computeTextureDimensions(volume, blockSize)

    if (verbose)
    {
        console.time('computePackedDistanceTexture')
    }

    const data = computePackedDistanceMaps(volume, variant, tolerance, blockSize, encoding, false)

    const texture = new THREE.Data3DTexture(data as any,dimensions.x,dimensions.y,dimensions.z)

    texture.format = textureFormat.format as any
    texture.type = textureFormat.type
    texture.internalFormat = textureFormat.internalFormat
    texture.minFilter = THREE.NearestFilter
    texture.magFilter = THREE.NearestFilter
    texture.generateMipmaps = false
    texture.unpackAlignment = 1
    texture.needsUpdate = true

    if (verbose)
    {
        console.timeEnd('computePackedDistanceTexture')
    }

    return { data, texture, dimensions, encoding }
}