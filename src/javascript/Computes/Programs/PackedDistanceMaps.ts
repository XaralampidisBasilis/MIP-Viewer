import * as tf from '@tensorflow/tfjs'
import { computeBidirectionalShadowMap } from './GPGPUShadowMapPaths'
import * as DistanceMap from './GPGPUDistanceMap'
import { type Axis, type Octant } from '../../Utils/ShadowMapUtils'
import {
    DISTANCE_TARGETS,
    maxDistance,
    packDistanceMaps,
    type DistanceMaps,
    type PackFormat,
    type PackedDistanceMaps,
} from './DistanceMapPackers'

export type DistanceKind = 'isotropic' | 'unidirectional' | 'bidirectional'

function computeDistance(
    shadowMap: tf.Tensor3D,
    kind: DistanceKind,
    axis: Axis,
    octant: Octant,
    maxDistance: number,
    verbose: boolean,
): tf.Tensor3D
{
    switch (kind)
    {
        case 'isotropic'     : return DistanceMap.computeIsotropicDistanceMap(shadowMap, maxDistance, verbose)
        case 'unidirectional': return DistanceMap.computeUnidirectionalDistanceMap(shadowMap, axis, octant, maxDistance, verbose)
        case 'bidirectional' : return DistanceMap.computeBidirectionalDistanceMap(shadowMap, axis, octant, maxDistance, verbose)
    }
}

export function computeDistanceArray(
    volume: tf.Tensor3D,
    kind: DistanceKind,
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
        distanceMap = computeDistance(shadowMap, kind, axis, octant, maxDistance, verbose)
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
    kind: DistanceKind,
    tolerance: number,
    blockSize: number,
    format: PackFormat,
    verbose: boolean = false,
): DistanceMaps<Int32Array>
{
    const maps = DISTANCE_TARGETS.map(({ axis, octant }) =>
        computeDistanceArray(volume, kind, axis, octant, tolerance, blockSize, maxDistance(format, axis), verbose)
    )

    return maps as DistanceMaps<Int32Array>
}

export function computePackedDistanceMaps(
    volume: tf.Tensor3D,
    kind: DistanceKind,
    tolerance: number,
    blockSize: number,
    format: PackFormat,
    verbose: boolean = false,
): PackedDistanceMaps
{
    return packDistanceMaps(
        computeDistanceMaps(volume, kind, tolerance, blockSize, format, verbose),
        format,
    )
}
