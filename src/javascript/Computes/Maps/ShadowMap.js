import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import * as S0 from '../Programs/GPGPUShadowMap'
import * as S1 from '../Programs/GPGPUShadowMapDifferences'

export default class ShadowMap
{
    constructor()
    {
        this.computes = new Computes()  
        this.configs = this.computes.configs
    }

    computeTensor()
    {
        console.time('computeTensor') 

        const volume = this.computes.volumeMap.tensor

        // this.tensor = S0.computeExtendedAnisotropicBidirectionalShadowMap(volume, 0.01, true)
        // this.tensor = S1.computeExtendedAnisotropicBidirectionalShadowMap(volume, 0.01, true)
        this.tensor = S1.computeExtendedAnisotropicBidirectionalBlockShadowMap(volume, 0.01, this.configs.blockSize, true)
        this.dimensions = new THREE.Vector3(...this.tensor.shape.toReversed())

        console.timeEnd('computeTensor') 
    }

    computeTexture()
    {
        console.time('computeTexture') 

        if (! this.textureData)
        {
            this.textureData = this.getTextureData()
        }

        this.texture = new THREE.Data3DTexture(this.textureData, ...this.dimensions)
        this.texture.format = THREE.RedIntegerFormat
        this.texture.type = THREE.ShortType
        this.texture.internalFormat = 'R16I'
        this.texture.minFilter = THREE.NearestFilter
        this.texture.magFilter = THREE.NearestFilter
        this.texture.generateMipmaps = false
        this.texture.unpackAlignment = 1
        this.texture.needsUpdate = true
        console.timeEnd('computeTexture') 
    }   

    updateTexture()
    {
        this.texture.image.data.set(this.getTextureData())
        this.texture.needsUpdate = true
    }

    computeTextureData()
    {
        this.textureData = new Int16Array(this.tensor.dataSync())
    }

    getTextureData()
    {
        return new Int16Array(this.tensor.dataSync())
    }

    dispose()
    {
        this.tensor?.dispose()
        this.texture?.dispose()
    }
}
