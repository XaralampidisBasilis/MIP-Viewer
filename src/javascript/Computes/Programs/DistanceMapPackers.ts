import { type Axis, type Octant, type Tuple } from '../../Utils/ShadowMapUtils'

export type PackFormat = '1bit' | '5bit' | '8bit' | '10bit'

export const DISTANCE_TARGETS: Tuple<{ axis: Axis, octant: Octant }, 12> = 
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

const MAX_DISTANCES: Record<PackFormat, Record<Axis, number>> = 
{
    '1bit' : { x: 1,    y: 1,    z: 1    },
    '5bit' : { x: 31,   y: 31,   z: 63   },
    '8bit' : { x: 255,  y: 255,  z: 255  },
    '10bit': { x: 2047, y: 2047, z: 1023 },
}

export function maxDistance(format: PackFormat, axis: Axis): number
{
    return MAX_DISTANCES[format][axis]
}

function voxelCount(maps: Tuple<Int32Array, 12>, name: string): number
{
    const voxels = maps[0].length

    for (let i = 1; i < maps.length; i++)
    {
        if (maps[i].length !== voxels)
        {
            throw new Error(`${name} expected map ${i} to have length ${voxels}, got ${maps[i].length}`)
        }
    }

    return voxels
}

function read4(
    maps: Tuple<Int32Array, 12>,
    offset: 0 | 4 | 8,
    voxel: number,
    mask: number,
): [number, number, number, number]
{
    return [
        maps[offset + 0][voxel] & mask,
        maps[offset + 1][voxel] & mask,
        maps[offset + 2][voxel] & mask,
        maps[offset + 3][voxel] & mask,
    ]
}

export function pack1Bit(maps: Tuple<Int32Array, 12>): Uint16Array
{
    const voxels = voxelCount(maps, 'pack1Bit')
    const packed = new Uint16Array(voxels)

    for (let i = 0; i < voxels; i++)
    {
        const x = read4(maps, 0, i, 0x1)
        const y = read4(maps, 4, i, 0x1)
        const z = read4(maps, 8, i, 0x1)

        packed[i] =
            (x[0] << 0) | (x[1] << 1) | (x[2] <<  2) | (x[3] <<  3) |
            (y[0] << 4) | (y[1] << 5) | (y[2] <<  6) | (y[3] <<  7) |
            (z[0] << 8) | (z[1] << 9) | (z[2] << 10) | (z[3] << 11)
    }

    return packed
}

export function pack5Bit(maps: Tuple<Int32Array, 12>): Uint16Array
{
    const voxels = voxelCount(maps, 'pack5Bit')
    const packed = new Uint16Array(voxels * 4)

    for (let i = 0; i < voxels; i++)
    {
        const x = read4(maps, 0, i, 0x1f)
        const y = read4(maps, 4, i, 0x1f)
        const z = read4(maps, 8, i, 0x3f)
        const i4 = i * 4

        packed[i4 + 0] = (x[0] | (y[0] << 5) | (z[0] << 10)) >>> 0
        packed[i4 + 1] = (x[1] | (y[1] << 5) | (z[1] << 10)) >>> 0
        packed[i4 + 2] = (x[2] | (y[2] << 5) | (z[2] << 10)) >>> 0
        packed[i4 + 3] = (x[3] | (y[3] << 5) | (z[3] << 10)) >>> 0
    }

    return packed
}

export function pack8Bit(maps: Tuple<Int32Array, 12>): Uint32Array
{
    const voxels = voxelCount(maps, 'pack8Bit')
    const packed = new Uint32Array(voxels * 3)

    for (let i = 0; i < voxels; i++)
    {
        const x = read4(maps, 0, i, 0xff)
        const y = read4(maps, 4, i, 0xff)
        const z = read4(maps, 8, i, 0xff)
        const i3 = i * 3

        packed[i3 + 0] = (x[0] | (x[1] << 8) | (x[2] << 16) | (x[3] << 24)) >>> 0
        packed[i3 + 1] = (y[0] | (y[1] << 8) | (y[2] << 16) | (y[3] << 24)) >>> 0
        packed[i3 + 2] = (z[0] | (z[1] << 8) | (z[2] << 16) | (z[3] << 24)) >>> 0
    }

    return packed
}

export function pack10Bit(maps: Tuple<Int32Array, 12>): Uint32Array
{
    const voxels = voxelCount(maps, 'pack10Bit')
    const packed = new Uint32Array(voxels * 4)

    for (let i = 0; i < voxels; i++)
    {
        const x = read4(maps, 0, i, 0x7ff)
        const y = read4(maps, 4, i, 0x7ff)
        const z = read4(maps, 8, i, 0x3ff)
        const i4 = i * 4

        packed[i4 + 0] = (x[0] | (y[0] << 11) | (z[0] << 22)) >>> 0
        packed[i4 + 1] = (x[1] | (y[1] << 11) | (z[1] << 22)) >>> 0
        packed[i4 + 2] = (x[2] | (y[2] << 11) | (z[2] << 22)) >>> 0
        packed[i4 + 3] = (x[3] | (y[3] << 11) | (z[3] << 22)) >>> 0
    }

    return packed
}

export function packDistanceMaps(maps: Tuple<Int32Array, 12>, format: PackFormat): Uint16Array | Uint32Array
{
    switch (format)
    {
        case '1bit' : return pack1Bit(maps)
        case '5bit' : return pack5Bit(maps)
        case '8bit' : return pack8Bit(maps)
        case '10bit': return pack10Bit(maps)
    }
}
