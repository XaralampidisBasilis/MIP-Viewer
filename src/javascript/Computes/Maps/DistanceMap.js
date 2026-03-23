import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import { toHalfFloat } from '../../Utils/DataUtils'
import * as GPGPU from '../Programs/GPGPUShadowDistanceMap'

export default class DistanceMap 
{
    constructor()
    {
        this.computes = new Computes()
        this.configs = this.computes.configs
    }
 
    computeR16UITexture()
    {
        console.time('computeR16UITexture') 

        // this.textureData = GPGPU.computeIsotropicDistanceMap1bit(this.computes.shadowMap.tensor, true)
        this.textureData = GPGPU.computeExtendedAnisotropicUnidirectionalDistanceMap1bit(this.computes.shadowMap.tensor, true)
        // this.textureData = GPGPU.computeExtendedAnisotropicBidirectionalDistanceMap1bit(this.computes.shadowMap.tensor, true)
        this.dimensions = this.computes.shadowMap.dimensions

        this.texture = new THREE.Data3DTexture(this.textureData, ...this.dimensions)
        this.texture.format = THREE.RedIntegerFormat
        this.texture.type = THREE.UnsignedShortType
        this.texture.internalFormat = 'R16UI'
        this.texture.minFilter = THREE.NearestFilter
        this.texture.magFilter = THREE.NearestFilter
        this.texture.generateMipmaps = false
        this.texture.unpackAlignment = 2
        this.texture.needsUpdate = true

        console.timeEnd('computeR16UITexture') 
    }   

    computeRGBA16UITexture()
    {
        console.time('computeRGBA16UITexture') 

        // this.textureData = GPGPU.computeIsotropicDistanceMap5bit(this.computes.shadowMap.tensor, true)
        this.textureData = GPGPU.computeExtendedAnisotropicUnidirectionalDistanceMap5bit(this.computes.shadowMap.tensor, true)
        // this.textureData = GPGPU.computeExtendedAnisotropicBidirectionalDistanceMap5bit(this.computes.shadowMap.tensor, true)
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

        // this.textureData = GPGPU.computeIsotropicDistanceMap8bit(this.computes.shadowMap.tensor, true)
        this.textureData = GPGPU.computeExtendedAnisotropicUnidirectionalDistanceMap8bit(this.computes.shadowMap.tensor, true)
        // this.textureData = GPGPU.computeExtendedAnisotropicBidirectionalDistanceMap8bit(this.computes.shadowMap.tensor, true)
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
