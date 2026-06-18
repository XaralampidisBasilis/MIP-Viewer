import * as THREE from 'three'
import { getWebGPUComputeContext } from '../../WebGPU/WebGPUDevice'
import { WebGPUTensor3D } from '../WebGPU/WebGPUTensor3D'
import {
    computeBidirectionalShadowMapWebGPU,
    computeUnidirectionalDistanceMapWebGPU,
} from '../WebGPU/WebGPUShadowDistanceKernels'

const DISTANCE_TARGETS =
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

const MAX_ENCODABLE_DISTANCES =
{
    '1bit' : { x: 1,    y: 1,    z: 1    },
    '5bit' : { x: 31,   y: 31,   z: 63   },
    '8bit' : { x: 255,  y: 255,  z: 255  },
    '10bit': { x: 2047, y: 2047, z: 1023 },
}

const DISTANCE_PACKERS =
{
    '1bit' : pack1Bit,
    '5bit' : pack5Bit,
    '8bit' : pack8Bit,
    '10bit': pack10Bit,
}

const DISTANCE_TEXTURE_FORMATS =
{
    '1bit' : { format: 'RED_INTEGER',  type: THREE.UnsignedShortType, internalFormat: 'R16UI'    },
    '5bit' : { format: 'RGBA_INTEGER', type: THREE.UnsignedShortType, internalFormat: 'RGBA16UI' },
    '8bit' : { format: 'RGB_INTEGER',  type: THREE.UnsignedIntType,   internalFormat: 'RGB32UI'  },
    '10bit': { format: 'RGBA_INTEGER', type: THREE.UnsignedIntType,   internalFormat: 'RGBA32UI' },
}

function maxEncodableDistance(encoding, axis)
{
    return MAX_ENCODABLE_DISTANCES[encoding][axis]
}

function computeTextureDimensionsFromShape(shape, blockSize)
{
    const blockShape = shape.map((x) => Math.ceil((x + 1) / blockSize))
    const [depth, height, width] = blockShape

    return new THREE.Vector3(width, height, depth)
}

function assertSameVoxelCount(maps, name)
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

function pack1Bit(maps)
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

function pack5Bit(maps)
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

function pack8Bit(maps)
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

function pack10Bit(maps)
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

async function ensureWebGPUTensor(volume, device)
{
    if (volume?.buffer && volume?.device && volume?.shape)
    {
        return { tensor: volume, owned: false }
    }

    if (!volume?.shape || typeof volume.data !== 'function')
    {
        throw new Error('computePackedDistanceTextureWebGPU expected a WebGPUTensor3D or tf.Tensor3D-like volume.')
    }

    const volumeData = await volume.data()
    const data = volumeData instanceof Float32Array ? volumeData : new Float32Array(volumeData)
    const tensor = WebGPUTensor3D.fromTypedArray(device, volume.shape, data, 'float32', 'webgpu-distance-volume')

    return { tensor, owned: true }
}

async function computeDistanceArray(volume, variant, axis, octant, tolerance, blockSize, maxDistance, verbose)
{
    if (variant !== 'unidirectional')
    {
        throw new Error(`WebGPU distance texture currently supports the active "unidirectional" variant, got "${variant}".`)
    }

    if (verbose)
    {
        console.time(`webgpu-distance-${axis}-${octant}`)
    }

    const shadowMap = await computeBidirectionalShadowMapWebGPU(volume, axis, octant, tolerance, blockSize)
    let distanceMap = null

    try
    {
        distanceMap = await computeUnidirectionalDistanceMapWebGPU(shadowMap, axis, octant, maxDistance)
        return await distanceMap.read()
    }
    finally
    {
        shadowMap.dispose()
        distanceMap?.dispose()

        if (verbose)
        {
            console.timeEnd(`webgpu-distance-${axis}-${octant}`)
        }
    }
}

async function computeDistanceMaps(volume, variant, tolerance, blockSize, encoding, verbose)
{
    const maps = []

    for (const { axis, octant } of DISTANCE_TARGETS)
    {
        const maxDistance = maxEncodableDistance(encoding, axis)
        const map = await computeDistanceArray(volume, variant, axis, octant, tolerance, blockSize, maxDistance, verbose)
        maps.push(map)
    }

    return maps
}

export async function computePackedDistanceTextureWebGPU(
    volume,
    variant,
    tolerance,
    blockSize,
    encoding,
    verbose = false,
)
{
    const { device } = await getWebGPUComputeContext()
    const { tensor, owned } = await ensureWebGPUTensor(volume, device)
    const textureFormat = DISTANCE_TEXTURE_FORMATS[encoding]
    const dimensions = computeTextureDimensionsFromShape(tensor.shape, blockSize)

    if (!textureFormat)
    {
        throw new Error(`Unsupported distance texture encoding "${encoding}".`)
    }

    if (verbose)
    {
        console.time('computePackedDistanceTextureWebGPU')
    }

    try
    {
        const maps = await computeDistanceMaps(tensor, variant, tolerance, blockSize, encoding, verbose)
        const data = DISTANCE_PACKERS[encoding](maps)
        const texture = new THREE.Data3DTexture(data, dimensions.x, dimensions.y, dimensions.z)

        texture.format = textureFormat.format
        texture.type = textureFormat.type
        texture.internalFormat = textureFormat.internalFormat
        texture.minFilter = THREE.NearestFilter
        texture.magFilter = THREE.NearestFilter
        texture.generateMipmaps = false
        texture.unpackAlignment = 1
        texture.needsUpdate = true

        return { data, texture, dimensions, encoding }
    }
    finally
    {
        if (owned)
        {
            tensor.dispose()
        }

        if (verbose)
        {
            console.timeEnd('computePackedDistanceTextureWebGPU')
        }
    }
}
