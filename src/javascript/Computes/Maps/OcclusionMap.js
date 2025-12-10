import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import { computeOcclusionMap } from '../Programs/GPGPUOcclusionMap'

export default class OcclusionMap
{
    constructor()
    {
        this.computes = new Computes()
        this.configs = this.computes.configs
        this.volumeMap = this.computes.volumeMap
    }

    computeTensor()
    {
        console.time('computeTensor') 
        this.tensor = computeOcclusionMap(this.volumeMap.tensor)
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
