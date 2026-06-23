import Computes from '../Computes'
import { computePackedDistanceTextureWebGPU } from '../WebGPU/WebGPUComputePackedDistanceTexture'

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

    async computeTexture()
    {
        this.distanceVariation = this.configs.distanceVariation
        this.errorTolerance = this.configs.errorTolerance
        this.blockSize = this.configs.blockSize

        const volume = this.computes.volumeMap.webgpuTensor

        const result = await computePackedDistanceTextureWebGPU(
            volume,
            'unidirectional',
            this.errorTolerance,
            this.blockSize,
            this.distanceVariation,
            this.configs.logComputeResults,
            { shadowBackend: this.configs.webgpuShadowBackend },
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
