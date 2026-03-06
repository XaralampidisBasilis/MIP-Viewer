import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import { toHalfFloat } from '../../Utils/DataUtils'
import { 
    computeExtendedAnisotropicBidirectionalShadowDistanceMap,
    computeExtendedAnisotropicBidirectionalShadowDistanceMapDebug,
} from '../Programs/GPGPUExtendedAnisotropicBidirectionalShadowDistanceMap'

export default class DistanceMap 
{
    constructor()
    {
        this.computes = new Computes()
        this.configs = this.computes.configs
        this.shadowMap = this.computes.shadowMap
        this.maxDistance = 31
    }

    computeTensor()
    {
        console.time('computeTensor') 
        this.dimensions = new THREE.Vector3(...this.shadowMap.dimensions)
        this.tensor = computeExtendedAnisotropicBidirectionalShadowDistanceMap(this.shadowMap.tensor, this.maxDistance)
        // this.tensor = computeExtendedAnisotropicBidirectionalShadowDistanceMapDebug(this.shadowMap.tensor, this.maxDistance, true)

        console.timeEnd('computeTensor') 
    }

    computeTexture()
    {
        console.time('computeTexture') 
        this.texture = new THREE.Data3DTexture(this.getTextureData(), ...this.dimensions)
        this.texture.format = THREE.RGBAIntegerFormat
        this.texture.type = THREE.UnsignedShortType
        this.texture.internalFormat = 'RGBA16UI'
        this.texture.minFilter = THREE.NearestFilter
        this.texture.magFilter = THREE.NearestFilter
        this.texture.generateMipmaps = false
        this.texture.unpackAlignment = 8
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
        const data = this.tensor.dataSync()

        try 
        {
            const f16 = new Float16Array(data)
            const u16 = new Uint16Array(f16.buffer)

            return u16
        }
        catch (error)
        {
            // Fallback if Float16Array is not supported
            const u16 = new Uint16Array(data.length)

            for (let i = 0; i < data.length; i++) 
            {
                u16[i] = toHalfFloat(data[i])
            }            
            return u16
        }
    }


    dispose()
    {
        this.tensor?.dispose()
        this.texture?.dispose()
    }
}
