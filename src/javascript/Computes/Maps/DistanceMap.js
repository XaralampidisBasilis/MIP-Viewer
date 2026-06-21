import Computes from '../Computes'
import { computePackedDistanceTexture } from '../Programs/ComputePackedDistanceTexture'
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

        const useWebGPU = this.configs.computeBackend === 'webgpu'
        const volume = useWebGPU ? this.computes.volumeMap.webgpuTensor : this.computes.volumeMap.tensor

        const compute = useWebGPU ? computePackedDistanceTextureWebGPU : computePackedDistanceTexture
        const result = await compute(
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
