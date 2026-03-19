import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import { toHalfFloat } from '../../Utils/DataUtils'
import { 
    computeIsotropicDistanceMap5bit,
    computeIsotropicDistanceMap8bit,
    computeExtendedAnisotropicUnidirectionalDistanceMap5bit,
    computeExtendedAnisotropicUnidirectionalDistanceMap8bit,
    computeExtendedAnisotropicBidirectionalDistanceMap5bit,
    computeExtendedAnisotropicBidirectionalDistanceMap8bit,
} from '../Programs/GPGPUShadowDistanceMap'

export default class DistanceMap 
{
    constructor()
    {
        this.computes = new Computes()
        this.configs = this.computes.configs
    }
 
    computeRGBA16UITexture()
    {
        console.time('computeRGBA16UITexture') 

        // this.textureData = computeIsotropicDistanceMapRGBA16UI(this.computes.shadowMap.tensor, true)
        this.textureData = computeExtendedAnisotropicUnidirectionalDistanceMap5bit(this.computes.shadowMap.tensor, true)
        // this.textureData = computeExtendedAnisotropicBidirectionalDistanceMap5bit(this.computes.shadowMap.tensor, true)
        this.dimensions = this.computes.shadowMap.dimensions

        this.texture = new THREE.Data3DTexture(this.textureData, ...this.dimensions)
        this.texture.format = THREE.RGBAIntegerFormat
        this.texture.type = THREE.UnsignedShortType
        this.texture.internalFormat = 'RGBA16UI'
        this.texture.minFilter = THREE.NearestFilter
        this.texture.magFilter = THREE.NearestFilter
        this.texture.generateMipmaps = false
        this.texture.unpackAlignment = 8
        this.texture.needsUpdate = true

        console.timeEnd('computeRGBA16UITexture') 
    }   

    computeRGB32UITexture()
    {
        console.time('computeRGB32UITexture') 

        // this.textureData = computeIsotropicDistanceMapRGB32UI(this.computes.shadowMap.tensor, true)
        this.textureData = computeExtendedAnisotropicUnidirectionalDistanceMap8bit(this.computes.shadowMap.tensor, true)
        // this.textureData = computeExtendedAnisotropicBidirectionalDistanceMap8bit(this.computes.shadowMap.tensor, true)
        this.dimensions = this.computes.shadowMap.dimensions

        // Workaround: this Three.js version does not map THREE.RGBIntegerFormat to WebGL's RGB_INTEGER
        // for Data3DTexture uploads, so use the raw WebGL enum name string instead.
        this.texture = new THREE.Data3DTexture(this.textureData, ...this.dimensions)
        this.texture.format = 'RGB_INTEGER' // THREE.RGBIntegerFormat
        this.texture.type = THREE.UnsignedIntType
        this.texture.internalFormat = 'RGB32UI'
        this.texture.minFilter = THREE.NearestFilter
        this.texture.magFilter = THREE.NearestFilter
        this.texture.generateMipmaps = false
        this.texture.unpackAlignment = 1
        this.texture.needsUpdate = true

        console.timeEnd('computeRGB32UITexture') 
    }   

    dispose()
    {
        this.texture?.dispose()
        this.textureData = null
    }
}
