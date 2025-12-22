import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import { maxPooling3d } from '../Programs/GPGPUMaxPooling'
import { computeExtendedAnisotropicOcclusionMap } from '../Programs/GPGPUExtendedAnisotropicOcclusionMap'

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

        const tensor = maxPooling3d(this.volumeMap.tensor, this.volumeMap.tensor.shape.map((x) => Math.ceil(x/2)))
        this.tensor = await computeExtendedAnisotropicOcclusionMap(tensor); tensor.dispose(0)
        console.log(this.tensor.mean().dataSync())
        
        const shape = this.tensor.shape
        this.dimensions = new THREE.Vector3(...shape.slice(0,3).toReversed())

        console.timeEnd('computeTensor') 
    }

    dispose()
    {
        this.tensor?.dispose()
        this.texture?.dispose()
    }
}
