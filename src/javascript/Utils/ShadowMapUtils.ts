export type Axis = 'x' | 'y' | 'z'
export type Sign = '+' | '-'
export type Dimension = 0 | 1 | 2
export type Octant = `${Sign}${Sign}${Sign}`
export type Permute = [Dimension, Dimension, Dimension]
export type Reverse = Dimension[]
export type Tuple<T, N extends number, R extends unknown[] = []> = R['length'] extends N ? R : Tuple<T, N, [T, ...R]>

const INDEX_FROM_AXIS: Record<Axis, Dimension> = 
{
    'x': 0,
    'y': 1,
    'z': 2,
}

const MAP_FROM_DOMINANT_AXIS_OCTANT: Record<string, number> = 
{
    "x|+++": 0,
    "x|---": 0,
    "x|+-+": 1,
    "x|-+-": 1,
    "x|--+": 2,
    "x|++-": 2,
    "x|-++": 3,
    "x|+--": 3,

    "y|+++": 4,
    "y|---": 4,
    "y|-++": 5,
    "y|+--": 5,
    "y|++-": 6,
    "y|--+": 6,
    "y|+-+": 7,
    "y|-+-": 7,

    "z|+++": 8,
    "z|---": 8,
    "z|+-+": 9,
    "z|-+-": 9,
    "z|-++": 10,
    "z|+--": 10,
    "z|++-": 11,
    "z|--+": 11,
}

const MAP_FROM_PERMUTE_REVERSE: Record<string, number> = 
{
    "2,1,0|"     : 0,
    "2,1,0|2,1,0": 0,
    "2,1,0|1"    : 1,
    "2,1,0|2,0"  : 1,
    "2,1,0|0"    : 2,
    "2,1,0|2,1"  : 2,
    "2,1,0|2"    : 3,
    "2,1,0|1,0"  : 3,

    "1,2,0|"     : 4,
    "1,2,0|1,2,0": 4,
    "1,2,0|2"    : 5,
    "1,2,0|1,0"  : 5,
    "1,2,0|0"    : 6,
    "1,2,0|1,2"  : 6,
    "1,2,0|1"    : 7,
    "1,2,0|2,0"  : 7,

    "0,1,2|"     : 8,
    "0,1,2|0,1,2": 8,
    "0,1,2|1"    : 9,
    "0,1,2|0,2"  : 9,
    "0,1,2|2"    : 10,
    "0,1,2|0,1"  : 10,
    "0,1,2|0"    : 11,
    "0,1,2|1,2"  : 11,
}

const PERMUTE_REVERSE_FROM_DOMINANT_AXIS_OCTANT: Record<string, { permute: Permute, reverse: Reverse }> = 
{
    "x|+++": { permute: [2,1,0], reverse: [] },
    "x|+-+": { permute: [2,1,0], reverse: [1] },
    "x|++-": { permute: [2,1,0], reverse: [0] },
    "x|+--": { permute: [2,1,0], reverse: [1,0] },
    "x|---": { permute: [2,1,0], reverse: [2,1,0] },
    "x|-+-": { permute: [2,1,0], reverse: [2,0] },
    "x|--+": { permute: [2,1,0], reverse: [2,1] },
    "x|-++": { permute: [2,1,0], reverse: [2] },

    "y|+++": { permute: [1,2,0], reverse: [] },
    "y|-++": { permute: [1,2,0], reverse: [2] },
    "y|++-": { permute: [1,2,0], reverse: [0] },
    "y|-+-": { permute: [1,2,0], reverse: [2,0] },
    "y|---": { permute: [1,2,0], reverse: [1,2,0] },
    "y|+--": { permute: [1,2,0], reverse: [1,0] },
    "y|--+": { permute: [1,2,0], reverse: [1,2] },
    "y|+-+": { permute: [1,2,0], reverse: [1] },

    "z|+++": { permute: [0,1,2], reverse: [] },
    "z|+-+": { permute: [0,1,2], reverse: [1] },
    "z|-++": { permute: [0,1,2], reverse: [2] },
    "z|--+": { permute: [0,1,2], reverse: [1,2] },
    "z|---": { permute: [0,1,2], reverse: [0,1,2] },
    "z|-+-": { permute: [0,1,2], reverse: [0,2] },
    "z|+--": { permute: [0,1,2], reverse: [0,1] },
    "z|++-": { permute: [0,1,2], reverse: [0] },
}

const DOMINANT_AXIS_OCTANT_FROM_PERMUTE_REVERSE: Record<string, { dominantAxis: Axis, octant: Octant }> = 
{
    "2,1,0|"     : { dominantAxis: 'x', octant: '+++' },
    "2,1,0|1"    : { dominantAxis: 'x', octant: '+-+' },
    "2,1,0|0"    : { dominantAxis: 'x', octant: '++-' },
    "2,1,0|1,0"  : { dominantAxis: 'x', octant: '+--' },
    "2,1,0|2,1,0": { dominantAxis: 'x', octant: '---' },
    "2,1,0|2,0"  : { dominantAxis: 'x', octant: '-+-' },
    "2,1,0|2,1"  : { dominantAxis: 'x', octant: '--+' },
    "2,1,0|2"    : { dominantAxis: 'x', octant: '-++' },

    "1,2,0|"     : { dominantAxis: 'y', octant: '+++' },
    "1,2,0|2"    : { dominantAxis: 'y', octant: '-++' },
    "1,2,0|0"    : { dominantAxis: 'y', octant: '++-' },
    "1,2,0|2,0"  : { dominantAxis: 'y', octant: '-+-' },
    "1,2,0|1,2,0": { dominantAxis: 'y', octant: '---' },
    "1,2,0|1,0"  : { dominantAxis: 'y', octant: '+--' },
    "1,2,0|1,2"  : { dominantAxis: 'y', octant: '--+' },
    "1,2,0|1"    : { dominantAxis: 'y', octant: '+-+' },

    "0,1,2|"     : { dominantAxis: 'z', octant: '+++' },
    "0,1,2|1"    : { dominantAxis: 'z', octant: '+-+' },
    "0,1,2|2"    : { dominantAxis: 'z', octant: '-++' },
    "0,1,2|1,2"  : { dominantAxis: 'z', octant: '--+' },
    "0,1,2|0,1,2": { dominantAxis: 'z', octant: '---' },
    "0,1,2|0,2"  : { dominantAxis: 'z', octant: '-+-' },
    "0,1,2|0,1"  : { dominantAxis: 'z', octant: '+--' },
    "0,1,2|0"    : { dominantAxis: 'z', octant: '++-' },
}

export function complementReverse(reverse: Reverse): Reverse
{
    const set = new Set<Dimension>(reverse)
    const complement: Reverse = []

    for (const axis of [0, 1, 2] as const)
    {
        if (!set.has(axis)) complement.push(axis)
    }
    return complement
}

export function inversePermutation(permute: Permute): Permute
{
    const inv = new Array<number>(permute.length)
    for (let i = 0; i < permute.length; i++)
    {
        inv[permute[i]] = i
    }

    return inv as Permute
}

export function applyPermutation(newOffset: [number, number, number], permute: Permute): [number, number, number]
{
    const oldOffset: [number, number, number] = [0, 0, 0]

    oldOffset[permute[0]] = newOffset[0]
    oldOffset[permute[1]] = newOffset[1]
    oldOffset[permute[2]] = newOffset[2]

    return oldOffset
}

export function axisIndex(axis: Axis): Dimension
{
    return INDEX_FROM_AXIS[axis]
}

export function reverseSign(sign: Sign): Sign
{
    return sign === '+' ? '-' : '+'
}

export function signFromOctant(octant: Octant, axis: Dimension): Sign
{
    return octant[axis] as Sign
}

export function reverseOctant(octant: Octant): Octant
{
    return `${reverseSign(signFromOctant(octant, 0))}${reverseSign(signFromOctant(octant, 1))}${reverseSign(signFromOctant(octant, 2))}`
}

export function mapFromPermuteReverse(permute: Permute, reverse: Reverse): number
{
    const key = `${permute.join(",")}|${reverse.join(",")}`
    const map = MAP_FROM_PERMUTE_REVERSE[key]

    if (map === undefined)
    {
        throw new Error("No mapping for " + key)
    }

    return map
}

export function mapFromDominantAxisOctant(dominantAxis: Axis, octant: Octant): number
{
    const key = `${dominantAxis}|${octant}`
    const map = MAP_FROM_DOMINANT_AXIS_OCTANT[key]

    if (map === undefined)
    {
        throw new Error("No mapping for " + key)
    }

    return map
}

export function permuteReverseFromDominantAxisOctant(dominantAxis: Axis, octant: Octant): { permute: Permute, reverse: Reverse }
{
    const key = `${dominantAxis}|${octant}`
    const v = PERMUTE_REVERSE_FROM_DOMINANT_AXIS_OCTANT[key]

    if (!v)
    {
        throw new Error("No mapping for " + key)
    }

    return { permute: [...v.permute] as Permute, reverse: [...v.reverse] as Reverse }
}

export function dominantAxisOctantFromPermuteReverse(permute: Permute, reverse: Reverse): { dominantAxis: Axis, octant: Octant }
{
    const key = `${permute.join(",")}|${reverse.join(",")}`
    const v = DOMINANT_AXIS_OCTANT_FROM_PERMUTE_REVERSE[key]

    if (!v)
    {
        throw new Error("No inverse mapping for " + key)
    }

    return { dominantAxis: v.dominantAxis, octant: v.octant }
}
