import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import { computeExtendedAnisotropicBidirectionalOcclusionMap } from '../Programs/GPGPUOcclusionMap'

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
        this.tensor = computeExtendedAnisotropicBidirectionalOcclusionMap(this.volumeMap.tensor)
        this.dimensions = new THREE.Vector3(...this.tensor.shape.slice(0,3).toReversed())
        console.timeEnd('computeTensor') 

        // const t1 = occ0.computeUnidirectionalOcclusionMapBase(this.volumeMap.tensor, [0,1,2], [])
        // const t2 = occ1.computeExtendedAnisotropicBidirectionalOcclusionMap(this.volumeMap.tensor, true)
        // const t3 = occ1.computeUnidirectionalOcclusionMapBase(this.volumeMap.tensor, [0,1,2], [], true)
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
