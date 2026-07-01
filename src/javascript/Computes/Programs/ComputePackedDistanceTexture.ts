import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import * as DistanceMap from './GPGPUDistanceMap'
import { computeBidirectionalShadowMap } from './GPGPUShadowMapFaces'
// import { computeBidirectionalShadowMap } from './GPGPUShadowMapPaths'
import { type Axis, type Octant, type Tuple } from '../../Utils/ShadowMapUtils'

type DistanceVariant = 'isotropic' | 'unidirectional' | 'bidirectional'
type DistanceEncoding = '1bit' | '5bit' | '8bit' | '10bit'
type DistanceMaps = Tuple<Int32Array, 12>
type PackedDistanceMaps = Uint16Array | Uint32Array
type MaxDistances = Record<Axis, number>
type PackedDistanceTexture =
{
    data: PackedDistanceMaps
    texture: THREE.Data3DTexture
    dimensions: THREE.Vector3
    encoding: DistanceEncoding
}
type DistanceTextureFormat =
{
    format: string
    type: number
    internalFormat: string
}

const DISTANCE_TARGETS: Tuple<{ axis: Axis, octant: Octant }, 12> =
[
    { axis: 'x', octant: '+++' },
    { axis: 'x', octant: '+-+' },
    { axis: 'x', octant: '++-' },
    { axis: 'x', octant: '+--' },

    { axis: 'y', octant: '+++' },
    { axis: 'y', octant: '-++' },
    { axis: 'y', octant: '++-' },
    { axis: 'y', octant: '-+-' },

    { axis: 'z', octant: '+++' },
    { axis: 'z', octant: '+-+' },
    { axis: 'z', octant: '-++' },
    { axis: 'z', octant: '--+' },
]

const MAX_ENCODABLE_DISTANCES: Record<DistanceEncoding, MaxDistances> =
{
    '1bit' : { x: 1,    y: 1,    z: 1    },
    '5bit' : { x: 31,   y: 31,   z: 63   },
    '8bit' : { x: 255,  y: 255,  z: 255  },
    '10bit': { x: 2047, y: 2047, z: 1023 },
}

const DISTANCE_PACKERS: Record<DistanceEncoding, (maps: DistanceMaps) => PackedDistanceMaps> =
{
    '1bit' : pack1Bit,
    '5bit' : pack5Bit,
    '8bit' : pack8Bit,
    '10bit': pack10Bit,
}

// This THREE version does not map THREE.RGBIntegerFormat to WebGL's RGB_INTEGER
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

function packDistanceMaps(maps: DistanceMaps, encoding: DistanceEncoding): PackedDistanceMaps
{
    return DISTANCE_PACKERS[encoding](maps)
}

function computeTextureDimensions(volume: tf.Tensor3D, blockSize: number): THREE.Vector3
{
    const blockShape = volume.shape.map((x) => Math.ceil((x + 1) / blockSize))
    const [depth, height, width] = blockShape

    return new THREE.Vector3(width, height, depth)
}

function assertSameVoxelCount(maps: DistanceMaps, name: string): number
{
    const voxels = maps[0].length

    for (let i = 1; i < maps.length; i += 1)
    {
        if (maps[i].length !== voxels)
        {
            throw new Error(`${name} expected map ${i} to have length ${voxels}, got ${maps[i].length}`)
        }
    }

    return voxels
}

function pack1Bit(maps: DistanceMaps): Uint16Array
{
    const voxels = assertSameVoxelCount(maps, 'pack1Bit')
    const packed = new Uint16Array(voxels)

    for (let i = 0; i < voxels; i += 1)
    {
        const x0 = maps[0][i] & 0x1
        const x1 = maps[1][i] & 0x1
        const x2 = maps[2][i] & 0x1
        const x3 = maps[3][i] & 0x1

        const y0 = maps[4][i] & 0x1
        const y1 = maps[5][i] & 0x1
        const y2 = maps[6][i] & 0x1
        const y3 = maps[7][i] & 0x1

        const z0 = maps[ 8][i] & 0x1
        const z1 = maps[ 9][i] & 0x1
        const z2 = maps[10][i] & 0x1
        const z3 = maps[11][i] & 0x1

        packed[i] =
        (x0 << 0) | (x1 << 1) | (x2 <<  2) | (x3 <<  3) |
        (y0 << 4) | (y1 << 5) | (y2 <<  6) | (y3 <<  7) |
        (z0 << 8) | (z1 << 9) | (z2 << 10) | (z3 << 11)
    }

    return packed
}

function pack5Bit(maps: DistanceMaps): Uint16Array
{
    const voxels = assertSameVoxelCount(maps, 'pack5Bit')
    const packed = new Uint16Array(voxels * 4)

    for (let i = 0; i < voxels; i += 1)
    {
        const x0 = maps[0][i] & 0x1f
        const x1 = maps[1][i] & 0x1f
        const x2 = maps[2][i] & 0x1f
        const x3 = maps[3][i] & 0x1f

        const y0 = maps[4][i] & 0x1f
        const y1 = maps[5][i] & 0x1f
        const y2 = maps[6][i] & 0x1f
        const y3 = maps[7][i] & 0x1f

        const z0 = maps[ 8][i] & 0x3f
        const z1 = maps[ 9][i] & 0x3f
        const z2 = maps[10][i] & 0x3f
        const z3 = maps[11][i] & 0x3f

        const i4 = i * 4

        packed[i4 + 0] = (x0 | (y0 << 5) | (z0 << 10)) >>> 0
        packed[i4 + 1] = (x1 | (y1 << 5) | (z1 << 10)) >>> 0
        packed[i4 + 2] = (x2 | (y2 << 5) | (z2 << 10)) >>> 0
        packed[i4 + 3] = (x3 | (y3 << 5) | (z3 << 10)) >>> 0
    }

    return packed
}

function pack8Bit(maps: DistanceMaps): Uint32Array
{
    const voxels = assertSameVoxelCount(maps, 'pack8Bit')
    const packed = new Uint32Array(voxels * 3)

    for (let i = 0; i < voxels; i += 1)
    {
        const x0 = maps[0][i] & 0xff
        const x1 = maps[1][i] & 0xff
        const x2 = maps[2][i] & 0xff
        const x3 = maps[3][i] & 0xff

        const y0 = maps[4][i] & 0xff
        const y1 = maps[5][i] & 0xff
        const y2 = maps[6][i] & 0xff
        const y3 = maps[7][i] & 0xff

        const z0 = maps[ 8][i] & 0xff
        const z1 = maps[ 9][i] & 0xff
        const z2 = maps[10][i] & 0xff
        const z3 = maps[11][i] & 0xff

        const i3 = i * 3

        packed[i3 + 0] = (x0 | (x1 << 8) | (x2 << 16) | (x3 << 24)) >>> 0
        packed[i3 + 1] = (y0 | (y1 << 8) | (y2 << 16) | (y3 << 24)) >>> 0
        packed[i3 + 2] = (z0 | (z1 << 8) | (z2 << 16) | (z3 << 24)) >>> 0
    }

    return packed
}

function pack10Bit(maps: DistanceMaps): Uint32Array
{
    const voxels = assertSameVoxelCount(maps, 'pack10Bit')
    const packed = new Uint32Array(voxels * 4)

    for (let i = 0; i < voxels; i += 1)
    {
        const x0 = maps[0][i] & 0x7ff
        const x1 = maps[1][i] & 0x7ff
        const x2 = maps[2][i] & 0x7ff
        const x3 = maps[3][i] & 0x7ff

        const y0 = maps[4][i] & 0x7ff
        const y1 = maps[5][i] & 0x7ff
        const y2 = maps[6][i] & 0x7ff
        const y3 = maps[7][i] & 0x7ff

        const z0 = maps[ 8][i] & 0x3ff
        const z1 = maps[ 9][i] & 0x3ff
        const z2 = maps[10][i] & 0x3ff
        const z3 = maps[11][i] & 0x3ff

        const i4 = i * 4

        packed[i4 + 0] = (x0 | (y0 << 11) | (z0 << 22)) >>> 0
        packed[i4 + 1] = (x1 | (y1 << 11) | (z1 << 22)) >>> 0
        packed[i4 + 2] = (x2 | (y2 << 11) | (z2 << 22)) >>> 0
        packed[i4 + 3] = (x3 | (y3 << 11) | (z3 << 22)) >>> 0
    }

    return packed
}

function computeDistance(
    shadowMap: tf.Tensor3D,
    variant: DistanceVariant,
    axis: Axis,
    octant: Octant,
    maxDistance: number,
    verbose: boolean,
): tf.Tensor3D
{
    switch (variant)
    {
        case 'isotropic':
            return DistanceMap.computeIsotropicDistanceMap(shadowMap, maxDistance, verbose)

        case 'unidirectional':
            return DistanceMap.computeUnidirectionalDistanceMap(shadowMap, axis, octant, maxDistance, verbose)

        case 'bidirectional':
            return DistanceMap.computeBidirectionalDistanceMap(shadowMap, axis, octant, maxDistance, verbose)
    }
}

export function computeDistanceArray(
    volume: tf.Tensor3D,
    variant: DistanceVariant,
    axis: Axis,
    octant: Octant,
    tolerance: number,
    blockSize: number,
    maxDistance: number,
    verbose: boolean = false,
): Int32Array
{
    const shadowMap = computeBidirectionalShadowMap(volume, axis, octant, tolerance, blockSize, false)
    let distanceMap: tf.Tensor3D | undefined

    try
    {
        distanceMap = computeDistance(shadowMap, variant, axis, octant, maxDistance, verbose)
        return distanceMap.dataSync() as Int32Array
    }
    finally
    {
        shadowMap.dispose()
        distanceMap?.dispose()
    }
}

export function computeDistanceMaps(
    volume: tf.Tensor3D,
    variant: DistanceVariant,
    tolerance: number,
    blockSize: number,
    encoding: DistanceEncoding,
    verbose: boolean = false,
): DistanceMaps
{
    const maps = DISTANCE_TARGETS.map(({ axis, octant }) => 
    {
        let maxDistance = maxEncodableDistance(encoding, axis)
        return computeDistanceArray(volume, variant, axis, octant, tolerance, blockSize, maxDistance, verbose)
    })

    return maps as DistanceMaps
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
    const maps = computeDistanceMaps(volume, variant, tolerance, blockSize, encoding, verbose)

    return packDistanceMaps(maps, encoding)
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

    const data = computePackedDistanceMaps( volume, variant, tolerance, blockSize, encoding, verbose)
 
    const texture = new THREE.Data3DTexture(data as any, dimensions.x, dimensions.y, dimensions.z)
    texture.format = textureFormat.format as any
    texture.type = textureFormat.type as any 
    texture.internalFormat = textureFormat.internalFormat as any
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