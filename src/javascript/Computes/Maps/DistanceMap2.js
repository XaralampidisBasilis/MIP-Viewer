import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import { toHalfFloat } from '../../Utils/DataUtils'
import { 
    computeExtendedAnisotropicBidirectionalDistanceMapStack,
} from '../Programs/GPGPUExtendedAnisotropicBidirectionalShadowDistanceMap'

export default class DistanceMap 
{
    constructor()
    {
        this.computes = new Computes()
        this.configs = this.computes.configs
        this.maxDistance = 255
    }

    computeTensor()
    {
        console.time('computeTensor') 
        
        this.tensor = computeExtendedAnisotropicBidirectionalDistanceMapStack(this.computes.shadowMap.tensor, this.maxDistance, true)

        const [depth, height, width] = this.tensor.shape
        this.dimensions = new THREE.Vector3(width, height, depth)

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
        this.texture.type = THREE.UnsignedByteType
        this.texture.internalFormat = 'R8UI'
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
        const data = this.tensor.dataSync()
        this.textureData = new Uint8Array(data)
    }

    getTextureData()
    {
        const data = this.tensor.dataSync()
        return new Uint8Array(data)
    }

    dispose()
    {
        this.tensor?.dispose()
        this.texture?.dispose()
    }
}
