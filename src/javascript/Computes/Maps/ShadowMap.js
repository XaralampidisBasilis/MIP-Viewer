import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import * as S0 from '../Programs/GPGPUShadowMap'
import * as S1 from '../Programs/GPGPUShadowMapDifferences'

import { computeShadowDistanceMapExtendedAnisotropicBidirectional, computeExtendedAnisotropicBidirectionalDistanceMap } from '../Programs/GPGPUExtendedAnisotropicBidirectionalShadowDistanceMap'

export default class ShadowMap
{
    constructor()
    {
        this.computes = new Computes()  
        this.configs = this.computes.configs
        this.volumeMap = this.computes.volumeMap
    }

    computeTensor()
    {
        console.time('computeTensor') 
        // this.tensor = S0.computeExtendedAnisotropicBidirectionalShadowMap(this.volumeMap.tensor, 0)
        this.tensor = S1.computeExtendedAnisotropicBidirectionalShadowMap(this.volumeMap.tensor, 0.01, true)
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
        this.texture.unpackAlignment = 2
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
