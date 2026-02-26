import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import * as S0 from '../Programs/GPGPUShadowMapOld'
import * as S1 from '../Programs/GPGPUShadowMap'
import * as S2 from '../Programs/GPGPUShadowMapDiff'

export default class OcclusionMap
{
    constructor()
    {
        this.computes = new Computes()  
        this.configs = this.computes.configs
        this.volumeMap = this.computes.volumeMap
    }

    async computeTensor()
    {
        console.time('computeTensor') 
        // tf.tidy(() => S0.computeExtendedAnisotropicBidirectionalShadowMap(this.volumeMap.tensor, true))
        // tf.tidy(() => S2.computeExtendedAnisotropicBidirectionalShadowMap(this.volumeMap.tensor, true))
        this.tensor = S2.computeExtendedAnisotropicUnidirectionalShadowMap(this.volumeMap.tensor, true)
        this.dimensions = new THREE.Vector3(...this.tensor.shape.toReversed())
        console.timeEnd('computeTensor') 
    }

    computeTexture()
    {
        console.time('computeTexture') 
        this.texture = new THREE.Data3DTexture(this.getTextureData(), ...this.dimensions)
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
