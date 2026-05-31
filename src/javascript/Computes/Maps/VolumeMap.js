import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import { resizeTrilinear } from '../Programs/resizeTrillinear'
import { resizeNearestNeighbor } from '../Programs/resizeNearestNeighbor'
import { map } from '../Programs/map_packed'
import { toHalfFloat } from '../../Utils/DataUtils'
import * as TensorUtils from '../../Utils/TensorUtils'

export default class VolumeMap
{
    constructor()
    {
        this.computes = new Computes()
        this.configs = this.computes.configs
        this.resources = this.computes.resources
    }

    setVolume()
    {
        this.volume = this.resources.items.volume
        this.shape = this.volume.dimensions.toReversed()
        this.dimensions = new THREE.Vector3().fromArray(this.volume.dimensions)
        this.spacing = new THREE.Vector3().fromArray(this.volume.spacing)
        this.size = new THREE.Vector3().fromArray(this.volume.size)
    }

    computeMinMax()
    {
        console.time('computeMinMax') 
        const minValue = tf.min(this.tensor)
        const maxValue = tf.max(this.tensor)

        this.minValue = minValue.arraySync()
        this.maxValue = maxValue.arraySync()

        tf.dispose([minValue, maxValue])
        console.timeEnd('computeMinMax') 
    }

    normalizeTensor()
    {
        console.time('normalizeTensor') 
        const mapped = map(this.tensor, this.minValue, this.maxValue)
        this.tensor.dispose()
        this.tensor = mapped  
        console.timeEnd('normalizeTensor')  
    }

    resizeTensor()
    {
        console.time('resizeTensor') 

        const shape = this.volume.dimensions.toReversed()
        const spacing = this.volume.spacing.toReversed()

        const newShape = shape.map((x) => Math.ceil(this.configs.downscaleFactor * x))
        const newSpacing = spacing.map((x, i) => shape[i]/newShape[i] * x)
        const resized = resizeTrilinear(this.tensor, newShape, false, true)
        this.tensor.dispose()

        this.shape = newShape
        this.dimensions.fromArray(newShape.toReversed())
        this.spacing.fromArray(newSpacing.toReversed())
        this.tensor = resized

        console.timeEnd('resizeTensor') 
    }

    computeTensor()
    {
        console.time('computeTensor') 
        
        this.setVolume()
        const shape = this.volume.dimensions.toReversed()
        const data = new Float32Array(this.volume.data)
        this.tensor = tf.tensor3d(data, shape)

        this.computeMinMax()
        this.normalizeTensor()

        if (this.configs.downscaleEnabled)
        {
            this.resizeTensor()
        }

        this.tensor.dispose()
        // this.tensor = TensorUtils.makeCartesianPlanes3d(this.shape, true)
        // this.tensor = TensorUtils.makeCartesianAxes3d(this.shape, true)
        this.tensor = TensorUtils.makeBoundaryPlanes3d(this.shape, true)

        console.log(this)
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
        this.texture.format = THREE.RedFormat
        this.texture.type = THREE.HalfFloatType
        this.texture.internalFormat = 'R16F'
        this.texture.minFilter = THREE.LinearFilter
        this.texture.magFilter = THREE.LinearFilter
        this.texture.generateMipmaps = false
        this.texture.needsUpdate = true
        this.texture.unpackAlignment = 2
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
        
        try 
        {
            const f16 = new Float16Array(data)
            this.textureData = new Uint16Array(f16.buffer)
        }
        catch (error)
        {
            this.textureData = new Uint16Array(data.length)

            for (let i = 0; i < data.length; i++) 
            {
                this.textureData[i] = toHalfFloat(data[i])
            }
        }
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
            const f16 = new Uint16Array(data.length)

            for (let i = 0; i < data.length; i++) 
            {
                f16[i] = toHalfFloat(data[i])
            }
            return f16
        }
    }

    dispose()
    {
        this.tensor?.dispose()
        this.texture?.dispose()
    }
}
