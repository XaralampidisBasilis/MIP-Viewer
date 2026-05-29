import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import * as GPGPU from '../Programs/GPGPUShadowDistanceMap'

import { computeVertexMargins as F1, computeVertexMinmax as G1 } from '../Programs/GPGPUShadowMapMargins'
import { computeVertexMargins as F2, computeVertexMinmax as G2 } from '../Programs/GPGPUShadowMapPaths'

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

        // const margins1 = F1(volume, [0,1,2], [])
        // const margins2 = F2(volume, [0,1,2], [])
        // const error = tf.abs(margins2.sub(margins1))

        // console.log('error z', error.mean([1,2,3,4]).mul(100).dataSync())
        // console.log('error y', error.mean([0,2,3,4]).mul(100).dataSync())
        // console.log('error x', error.mean([0,1,3,4]).mul(100).dataSync())

        // const minmax1 = G1(volume, [0,1,2], [])
        // const minmax2 = G2(volume, [0,1,2], [])
        // const error1 = tf.abs(minmax1.sub(volume))
        // const error2 = tf.abs(minmax2.sub(volume))
        
        // console.log('error 1: x = 0', error1.slice([0,0,0], [-1,-1,1]).mean().dataSync())
        // console.log('error 1: y = 0', error1.slice([0,0,0], [-1,1,-1]).mean().dataSync())
        // console.log('error 1: z = 0', error1.slice([0,0,0], [1,-1,-1]).mean().dataSync())

        // console.log('error 2: x = 0', error2.slice([1,0,0], [-1,-1,1]).mean().dataSync())
        // console.log('error 2: y = 0', error2.slice([0,0,0], [-1,1,-1]).mean().dataSync())
        // console.log('error 2: z = 0', error2.slice([0,0,0], [1,-1,-1]).mean().dataSync())

        if (this.distanceVariation ===  '1bit') this.compute1BitDistanceTexture()
        if (this.distanceVariation ===  '5bit') this.compute5BitDistanceTexture()
        if (this.distanceVariation ===  '8bit') this.compute8BitDistanceTexture()
        if (this.distanceVariation === '10bit') this.compute10BitDistanceTexture()
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
