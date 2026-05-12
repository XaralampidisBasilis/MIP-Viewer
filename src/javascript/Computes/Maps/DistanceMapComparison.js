import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import * as GPGPU from '../Programs/GPGPUShadowDistanceMapComparison'

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
        if (this.distanceVariation ===  '1bit') this.computeR16UITexture()
        if (this.distanceVariation ===  '5bit') this.computeRGBA16UITexture()
        if (this.distanceVariation ===  '8bit') this.computeRGB32UITexture()
        if (this.distanceVariation === '10bit') this.computeRGBA32UITexture()
    }
 
    computeR16UITexture()
    {
        console.time('computeR16UITexture@DistanceMapComparison') 
        
        const volumeTensor = this.computes.volumeMap.tensor
        const blockShape = volumeTensor.shape.map(X => Math.ceil((X + 1) / this.blockSize))
        
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

        console.timeEnd('computeR16UITexture@DistanceMapComparison') 
    }   

    computeRGBA16UITexture()
    {
        console.time('computeRGBA16UITexture@DistanceMapComparison') 

        const volumeTensor = this.computes.volumeMap.tensor
        const blockShape = volumeTensor.shape.map(X => Math.ceil((X + 1) / this.blockSize))

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

        console.timeEnd('computeRGBA16UITexture@DistanceMapComparison') 
    }   

    computeRGB32UITexture()
    {
        console.time('computeRGB32UITexture@DistanceMapComparison') 

        const volumeTensor = this.computes.volumeMap.tensor
        const blockShape = volumeTensor.shape.map(X => Math.ceil((X + 1) / this.blockSize))

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

        console.timeEnd('computeRGB32UITexture@DistanceMapComparison') 
    }   

    computeRGBA32UITexture()
    {
        console.time('computeRGBA32UITexture@DistanceMapComparison') 

        const volumeTensor = this.computes.volumeMap.tensor
        const blockShape = volumeTensor.shape.map(X => Math.ceil((X + 1) / this.blockSize))

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

        console.timeEnd('computeRGBA32UITexture@DistanceMapComparison') 
    }   

    dispose()
    {
        this.texture?.dispose()
        this.textureData = null
    }
}
