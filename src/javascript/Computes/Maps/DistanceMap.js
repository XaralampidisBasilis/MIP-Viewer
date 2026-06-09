import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import * as GPGPU from '../Programs/GPGPUShadowDistanceMap'

import * as A from '../Programs/GPGPUShadowMapPaths'
import * as B from '../Programs/GPGPUShadowMapPathsPacked'
import * as C from '../Programs/GPGPUShadowDistanceMap'

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

        tf.tidy(() => console.log(B.computeBidirectionalShadowMap(volume, 'z',   '+', 0.01, false).mean([0,1,2]).dataSync()))
        tf.tidy(() => console.log(A.computeBidirectionalShadowMap(volume, 'z', '+++', 0.01, false).mean([0,1,2]).dataSync()))
        tf.tidy(() => console.log(A.computeBidirectionalShadowMap(volume, 'z', '+-+', 0.01, false).mean([0,1,2]).dataSync()))
        tf.tidy(() => console.log(A.computeBidirectionalShadowMap(volume, 'z', '-++', 0.01, false).mean([0,1,2]).dataSync()))
        tf.tidy(() => console.log(A.computeBidirectionalShadowMap(volume, 'z', '--+', 0.01, false).mean([0,1,2]).dataSync()))

        tf.tidy(() => console.log(B.computeShadowDistanceMap(volume, 'z','+', 0.01, 1, 64, false).mean([0,1,2]).dataSync()))
        tf.tidy(() => C.extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '+++', 0.01, 1, 64, true))
        tf.tidy(() => C.extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '+-+', 0.01, 1, 64, true))
        tf.tidy(() => C.extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '-++', 0.01, 1, 64, true))
        tf.tidy(() => C.extendedAnisotropicUnidirectionalDistanceMapInt32Array(volume, 'z', '--+', 0.01, 1, 64, true))

        // if (this.distanceVariation ===  '1bit') this.compute1BitDistanceTexture()
        // if (this.distanceVariation ===  '5bit') this.compute5BitDistanceTexture()
        // if (this.distanceVariation ===  '8bit') this.compute8BitDistanceTexture()
        // if (this.distanceVariation === '10bit') this.compute10BitDistanceTexture()
    }
 
    compute1BitDistanceTexture()
    {
        console.time('computeR16UITexture') 
        
        const volumeTensor = this.computes.volumeMap.tensor
        const blockShape = volumeTensor.shape.map(x => Math.ceil((x + 1) / this.blockSize))

        this.dimensions = new THREE.Vector3().fromArray(blockShape.toReversed())
        this.textureData = GPGPU.computeExtendedAnisotropicUnidirectionalDistanceMap1bit(volumeTensor, this.errorTolerance, this.blockSize, true)

        this.texture = new THREE.Data3DTexture(this.textureData, ...this.dimensions)
        this.texture.format = 'RED_INTEGER' // THREE.RedIntegerFormat
        this.texture.type = THREE.UnsignedShortType
        this.texture.internalFormat = 'R16UI'
        this.texture.minFilter = THREE.NearestFilter
        this.texture.magFilter = THREE.NearestFilter
        this.texture.generateMipmaps = false
        this.texture.unpackAlignment = 1
        this.texture.needsUpdate = true

        console.timeEnd('computeR16UITexture') 
    }   

    compute5BitDistanceTexture()
    {
        console.time('computeRGBA16UITexture') 

        const volumeTensor = this.computes.volumeMap.tensor
        const blockShape = volumeTensor.shape.map(x => Math.ceil((x + 1) / this.blockSize))

        this.dimensions = new THREE.Vector3().fromArray(blockShape.toReversed())
        this.textureData = GPGPU.computeExtendedAnisotropicUnidirectionalDistanceMap5bit(volumeTensor, this.errorTolerance, this.blockSize, true)

        this.texture = new THREE.Data3DTexture(this.textureData, ...this.dimensions)
        this.texture.format = 'RGBA_INTEGER' // THREE.RGBAIntegerFormat
        this.texture.type = THREE.UnsignedShortType
        this.texture.internalFormat = 'RGBA16UI'
        this.texture.minFilter = THREE.NearestFilter
        this.texture.magFilter = THREE.NearestFilter
        this.texture.generateMipmaps = false
        this.texture.unpackAlignment = 1
        this.texture.needsUpdate = true

        console.timeEnd('computeRGBA16UITexture') 
    }   

    compute8BitDistanceTexture()
    {
        console.time('computeRGB32UITexture') 

        const volumeTensor = this.computes.volumeMap.tensor
        const blockShape = volumeTensor.shape.map(x => Math.ceil((x + 1) / this.blockSize))

        this.dimensions = new THREE.Vector3().fromArray(blockShape.toReversed())
        this.textureData = GPGPU.computeExtendedAnisotropicUnidirectionalDistanceMap8bit(volumeTensor, this.errorTolerance, this.blockSize, true)

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

    compute10BitDistanceTexture()
    {
        console.time('computeRGBA32UITexture') 

        const volumeTensor = this.computes.volumeMap.tensor
        const blockShape = volumeTensor.shape.map(x => Math.ceil((x + 1) / this.blockSize))

        this.dimensions = new THREE.Vector3().fromArray(blockShape.toReversed())
        this.textureData = GPGPU.computeExtendedAnisotropicUnidirectionalDistanceMap10bit(volumeTensor, this.errorTolerance, this.blockSize, true)

        // Workaround: this Three.js version does not map THREE.RGBIntegerFormat to WebGL's RGB_INTEGER
        // for Data3DTexture uploads, so use the raw WebGL enum name string instead.
        this.texture = new THREE.Data3DTexture(this.textureData, ...this.dimensions)
        this.texture.format = 'RGBA_INTEGER' // THREE.RGBAIntegerFormat
        this.texture.type = THREE.UnsignedIntType
        this.texture.internalFormat = 'RGBA32UI'
        this.texture.minFilter = THREE.NearestFilter
        this.texture.magFilter = THREE.NearestFilter
        this.texture.generateMipmaps = false
        this.texture.unpackAlignment = 1
        this.texture.needsUpdate = true

        console.timeEnd('computeRGBA32UITexture') 
    }   

    dispose()
    {
        this.texture?.dispose()
        this.textureData = null
    }
}
