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
        // console.log(await tf.profile(async () => await OCC1.computeExtendedAnisotropicBidirectionalOcclusionMapAsync(this.volumeMap.tensor) ))
        // const t1 = (await OCC.computeUnidirectionalOcclusionMap(this.volumeMap.tensor.transpose([1,0,2]))).transpose([1,0,2])
       
        // const t1 = OCC.computeUnidirectionalOcclusionMap(this.volumeMap.tensor, [1,0,2], [1])
        // const t2 = OCC1.computeUnidirectionalOcclusionMap(this.volumeMap.tensor, [1,0,2], [1])
        // const t3 = await OCC1.computeUnidirectionalOcclusionMapAsync(this.volumeMap.tensor, [1,0,2], [1])
        // tf.tidy(() => console.log(tf.sub(t1, t2).abs().mean().dataSync()[0]))
        // tf.tidy(() => console.log(tf.sub(t1, t3).abs().mean().dataSync()[0]))

        const t1 = OCC.computeBidirectionalOcclusionMap(this.volumeMap.tensor, [2,1,0], [])
        const t2 = await OCC.computeBidirectionalOcclusionMapAsync(this.volumeMap.tensor, [2,1,0], [])
        const t3 = await OCC1.computeBidirectionalOcclusionMapAsync(this.volumeMap.tensor, [2,1,0], [])
        tf.tidy(() => console.log(tf.sub(t1, t2).abs().mean().dataSync()[0]))
        tf.tidy(() => console.log(tf.sub(t1, t3).abs().mean().dataSync()[0]))

        // this.tensor = OCC.computeExtendedAnisotropicBidirectionalOcclusionMap(this.volumeMap.tensor)
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
