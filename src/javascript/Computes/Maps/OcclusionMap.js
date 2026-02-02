import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'

import * as OCC from '../Programs/GPGPUOcclusionMap'
import * as OCC1 from '../Programs/GPGPUOcclusionMap1'

export class OcclusionMap
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
        console.log(await tf.profile(() => tf.tidy(() => OCC.computeUnidirectionalOcclusionMap(this.volumeMap.tensor.transpose([2,1,0])).transpose([2,1,0]) )))
        console.log(await tf.profile(() => tf.tidy(() => OCC1.computeUnidirectionalOcclusionMap(this.volumeMap.tensor, [2,1,0], []))))
        // this.tensor = computeExtendedAnisotropicBidirectionalOcclusionMap(this.volumeMap.tensor)
        // this.dimensions = new THREE.Vector3(...this.tensor.shape.slice(0,3).toReversed())
        console.timeEnd('computeTensor') 
    }

    computeTexture()
    {
        console.time('computeTexture') 
        this.texture = new THREE.Data3DTexture(this.getTextureData(), ...this.dimensions)
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

    getTextureData()
    {
        return new Uint8Array(this.tensor.dataSync())
    }

    dispose()
    {
        this.tensor?.dispose()
        this.texture?.dispose()
    }
}
