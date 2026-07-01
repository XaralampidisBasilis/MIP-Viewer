import Computes from '../Computes'
import { computeDistanceMapsPackedTexture } from '../Programs/ComputePackedDistanceTexture'
// import { computePackedDistanceTexture } from '../Programs/ComputeFusedPackedDistanceTexture'

export default class DistanceMap
{
    constructor()
    {
        this.computes = new Computes()
        this.configs = this.computes.configs

        this.distanceVariation = this.configs.distanceVariation
        this.errorTolerance = this.configs.errorTolerance
        this.blockSize = this.configs.blockSize
    }

    computeTexture()
    {
        const volume = this.computes.volumeMap.tensor

        const result = computeDistanceMapsPackedTexture(
            volume,
            'unidirectional',
            this.errorTolerance,
            this.blockSize,
            this.distanceVariation,
            true,
        )

        this.textureData = result.data
        this.texture = result.texture
        this.dimensions = result.dimensions
    }

    dispose()
    {
        this.texture?.dispose()

        this.textureData = null
        this.texture = null
        this.dimensions = null
    }
}