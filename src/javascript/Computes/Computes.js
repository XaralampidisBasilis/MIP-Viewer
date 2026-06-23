import * as tf from '@tensorflow/tfjs'
import EventEmitter from '../Utils/EventEmitter'
import Experience from '../Experience'
import VolumeMap from './Maps/VolumeMap'
import DistanceMap from './Maps/DistanceMap'

export default class Computes extends EventEmitter
{
    static instance = null

    constructor()
    {
        super()

        if (Computes.instance) 
        {
            return Computes.instance
        }
        Computes.instance = this

        this.experience = new Experience()
        this.renderer = this.experience.renderer
        this.configs = this.experience.configs
        this.resources = this.experience.resources
        
        this.volumeMap = new VolumeMap()
        this.distanceMap = new DistanceMap()    
    }

    async start()
    {
        console.time('start@Computes') 

        await this.volumeMap.computeTensor()
        await tf.nextFrame()

        await this.distanceMap.computeTexture()
        await tf.nextFrame()

        await this.volumeMap.computeTexture()
        this.volumeMap.tensor?.dispose()
        this.volumeMap.tensor = null
        this.volumeMap.webgpuTensor?.dispose()
        this.volumeMap.webgpuTensor = null
        await tf.nextFrame()

        console.timeEnd('start@Computes') 
    }

    async change(event)
    {
        if      (event.key === 'blockSize'      ) await this.onChangeBlockSize(event)
        else if (event.key === 'downscaleFactor') await this.onChangeDownscaleFactor(event)
        else if (event.key === 'skippingMethod' ) await this.onChangeSkippingMethod(event)
    }

    async onChangeBlockSize(event)
    {
        console.time('onChangeBlockSize@Computes') 

        await this.volumeMap.computeTensor()
        await tf.nextFrame()

        await this.distanceMap.computeTexture()
        await tf.nextFrame()

        await this.volumeMap.computeTexture()
        this.volumeMap.tensor?.dispose()
        this.volumeMap.tensor = null
        this.volumeMap.webgpuTensor?.dispose()
        this.volumeMap.webgpuTensor = null
        await tf.nextFrame()

        console.timeEnd('onChangeBlockSize@Computes')
        this.printResources() 
    }

    async onChangeDownscaleFactor(event)
    {
        console.time('onChangeDownscaleFactor@Computes') 

        await this.volumeMap.computeTensor()
        await tf.nextFrame()

        await this.distanceMap.computeTexture()
        await tf.nextFrame()

        await this.volumeMap.computeTexture()
        this.volumeMap.tensor?.dispose()
        this.volumeMap.tensor = null
        this.volumeMap.webgpuTensor?.dispose()
        this.volumeMap.webgpuTensor = null
        await tf.nextFrame()

        console.timeEnd('onChangeDownscaleFactor@Computes') 
        this.printResources()
    }

    async onChangeSkippingMethod(event)
    {
        return
    }

    destroy()
    {
        this.volumeMap.dispose()
        this.distanceMap.dispose()

        this.volumeMap = null
        this.distanceMap = null

        this.experience = null
        this.renderer = null
        this.configs = null
        this.resources = null

        Computes.instance = null

        console.log('Computes destroyed')
    }
    
    printResources()
    {
        console.log(`Num of tensors: ${tf.memory().numTensors}, Num of textures: ${this.renderer.instance.info.memory.textures}`)
        console.log(``)
    }
}
