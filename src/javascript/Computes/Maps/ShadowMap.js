import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import * as S0 from '../Programs/GPGPUShadowMapOld'
import * as S1 from '../Programs/GPGPUShadowMap'

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
        this.tensor = S0.computeExtendedAnisotropicBidirectionalShadowMapDebug(this.volumeMap.tensor, true)
        this.dimensions = new THREE.Vector3(...this.tensor.shape.toReversed())

        // const t1 = S0.computeUnidirectionalShadowMap(this.volumeMap.tensor, [0,1,2], [], true)
        // const t2 = S0.computeBidirectionalShadowMap(this.volumeMap.tensor, [0,1,2], [], true)
        console.timeEnd('computeTensor') 
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

// 0.325439453125 = 0.408935546875
// 0.277099609375 = 0.394287109375
// 0.356689453125 = 0.4287109375
// 0.298095703125 = 0.380859375
// 0.43505859375  = 0.50048828125
// 0.431396484375 = 0.5048828125
// 0.446044921875 = 0.5244140625
// 0.425048828125 = 0.497314453125
// 0.401123046875 = 0.49951171875
// 0.40185546875  = 0.4931640625
// 0.399169921875 = 0.493408203125
// 0.409912109375 = 0.473388671875

