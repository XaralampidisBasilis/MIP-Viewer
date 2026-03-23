import * as THREE from 'three'
import Computes from '../Computes'
import * as GPGPU from '../Programs/GPGPUShadowDistanceMap2'

export default class DistanceMap2 
{
    constructor()
    {
        this.computes = new Computes()
        this.configs = this.computes.configs

        this.errorTolerance = this.configs.errorTolerance
        this.blockSize = this.configs.blockSize
    }
 
    computeR16UITexture()
    {
        console.time('computeR16UITexture') 
        
        const volumeTensor = this.computes.volumeMap.tensor
        const blockShape = volumeTensor.shape.map(x => Math.ceil((x + 1) / this.blockSize))
        this.dimensions = new THREE.Vector3().fromArray(blockShape.toReversed())

        // this.textureData = GPGPU.computeIsotropicDistanceMap1bit(volumeTensor, this.errorTolerance, this.blockSize, true)
        this.textureData = GPGPU.computeExtendedAnisotropicUnidirectionalDistanceMap1bit(volumeTensor, this.errorTolerance, this.blockSize, true)
        // this.textureData = GPGPU.computeExtendedAnisotropicBidirectionalDistanceMap1bit(volumeTensor, this.errorTolerance, this.blockSize, true)

        this.texture = new THREE.Data3DTexture(this.textureData, ...this.dimensions)
        this.texture.format = THREE.RedIntegerFormat
        this.texture.type = THREE.UnsignedShortType
        this.texture.internalFormat = 'R16UI'
        this.texture.minFilter = THREE.NearestFilter
        this.texture.magFilter = THREE.NearestFilter
        this.texture.generateMipmaps = false
        this.texture.unpackAlignment = 1
        this.texture.needsUpdate = true

        console.timeEnd('computeR16UITexture') 
    }   

    computeRGBA16UITexture()
    {
        console.time('computeRGBA16UITexture') 

        const volumeTensor = this.computes.volumeMap.tensor
        const blockShape = volumeTensor.shape.map(x => Math.ceil((x + 1) / this.blockSize))
        this.dimensions = new THREE.Vector3().fromArray(blockShape.toReversed())

        // this.textureData = GPGPU.computeIsotropicDistanceMap5bit(volumeTensor, this.errorTolerance, this.blockSize, true)
        this.textureData = GPGPU.computeExtendedAnisotropicUnidirectionalDistanceMap5bit(volumeTensor, this.errorTolerance, this.blockSize, true)
        // this.textureData = GPGPU.computeExtendedAnisotropicBidirectionalDistanceMap5bit(volumeTensor, this.errorTolerance, this.blockSize, true)

        this.texture = new THREE.Data3DTexture(this.textureData, ...this.dimensions)
        this.texture.format = THREE.RGBAIntegerFormat
        this.texture.type = THREE.UnsignedShortType
        this.texture.internalFormat = 'RGBA16UI'
        this.texture.minFilter = THREE.NearestFilter
        this.texture.magFilter = THREE.NearestFilter
        this.texture.generateMipmaps = false
        this.texture.unpackAlignment = 1
        this.texture.needsUpdate = true

        console.timeEnd('computeRGBA16UITexture') 
    }   

    computeRGB32UITexture()
    {
        console.time('computeRGB32UITexture') 

        const volumeTensor = this.computes.volumeMap.tensor
        const blockShape = volumeTensor.shape.map(x => Math.ceil((x + 1) / this.blockSize))
        this.dimensions = new THREE.Vector3().fromArray(blockShape.toReversed())

        // this.textureData = GPGPU.computeIsotropicDistanceMap8bit(volumeTensor, this.errorTolerance, this.blockSize, true)
        this.textureData = GPGPU.computeExtendedAnisotropicUnidirectionalDistanceMap8bit(volumeTensor, this.errorTolerance, this.blockSize, true)
        // this.textureData = GPGPU.computeExtendedAnisotropicBidirectionalDistanceMap8bit(volumeTensor, this.errorTolerance, this.blockSize, true)

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
