import * as THREE from 'three'
import * as tf from '@tensorflow/tfjs'
import Computes from '../Computes'
import { resizeTrilinear } from '../Programs/GPGPUResizeTrilinear'
import { normalizePacked } from '../Programs/normalize_packed'
import { mapPacked } from '../Programs/map_packed'

function makeCartesianPlanes(shape) 
{
    const [Z, Y, X] = shape;
    const midZ = Math.floor(Z / 2);
    const midY = Math.floor(Y / 2);
    const midX = Math.floor(X / 2);

    return tf.tidy(() => 
    {
        const z = tf.range(0, Z, 1, 'int32').reshape([Z, 1, 1]);
        const y = tf.range(0, Y, 1, 'int32').reshape([1, Y, 1]);
        const x = tf.range(0, X, 1, 'int32').reshape([1, 1, X]);

        const zPlane = z.equal(midZ).toFloat(); // [Z,1,1]
        const yPlane = y.equal(midY).toFloat(); // [1,Y,1]
        const xPlane = x.equal(midX).toFloat(); // [1,1,X]

        // broadcasting happens here
        const mask = tf.add(tf.add(zPlane, yPlane), xPlane);

        // intersections would be 2 or 3 -> clamp to 1
        return mask.clipByValue(0, 1);
    });
}

function makeCartesianPlanesRand(shape) 
{
    const [Z, Y, X] = shape;
    const midZ = Math.floor(Z / 2);
    const midY = Math.floor(Y / 2);
    const midX = Math.floor(X / 2);

    return tf.tidy(() => 
    {
        const z = tf.range(0, Z, 1, 'int32').reshape([Z, 1, 1]);
        const y = tf.range(0, Y, 1, 'int32').reshape([1, Y, 1]);
        const x = tf.range(0, X, 1, 'int32').reshape([1, 1, X]);

        const zPlane = z.equal(midZ).toFloat();
        const yPlane = y.equal(midY).toFloat();
        const xPlane = x.equal(midX).toFloat();

        const mask = tf.maximum(tf.maximum(zPlane, yPlane), xPlane); // [Z,Y,X] via broadcast

        const rnd = tf.randomUniform([Z, Y, X], 0, 1, 'float32');

        return rnd.mul(mask); // random on planes, 0 elsewhere
    });
}   

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
        this.dimensions = new THREE.Vector3().fromArray(this.volume.dimensions)
        this.spacing = new THREE.Vector3().fromArray(this.volume.spacing)
        this.size = new THREE.Vector3().fromArray(this.volume.size)
    }

    computeTensor()
    {
        console.time('computeTensor') 
        
        this.setVolume()
        this.downscaleFactor = this.configs.downscaleFactor

        this.tensor = tf.tidy(() =>
        {
            const shape = this.volume.dimensions.toReversed()
            const data = new Float32Array(this.volume.data)
            const tensor = tf.tensor3d(data, shape)

            this.minValue = tf.min(tensor).arraySync()
            this.maxValue = tf.max(tensor).arraySync()

            const mapped = mapPacked(tensor, this.minValue, this.maxValue)
            tensor.dispose()

            const newShape = this.volume.dimensions.toReversed().map((x) => Math.ceil(this.downscaleFactor * x))
            const newSpacing = this.volume.spacing.toReversed().map((x, i) => shape[i]/newShape[i] * x)

            this.dimensions.fromArray(newShape.toReversed())
            this.spacing.fromArray(newSpacing.toReversed())

            return resizeTrilinear(mapped, newShape, false, true)
        })  

        // this.tensor = tf.tidy(() => makeCartesianPlanesRand(this.tensor.shape))

        console.timeEnd('computeTensor') 
        console.log(this)
    }

    computeMipmap()
    {
        console.time('computeMipmap') 
        
        this.mipmap = {}
        this.mipmap.tensor = tf.tidy(() =>
        {
            const shape = this.tensor.shape.map(x => Math.ceil(x / this.mipmap.blockSize))
            return resizeTrilinear(this.tensor, shape, false, true)
        })  

        this.mipmap.blockSize = this.configs.blockSize
        this.mipmap.dimensions = new THREE.Vector3().fromArray(this.mipmap.shape.toReversed())

        console.timeEnd('computeMipmap') 
    }

    computeTexture()
    {
        console.time('computeTexture') 
        this.texture = new THREE.Data3DTexture(this.getTextureData(), ...this.dimensions)
        this.texture.format = THREE.RedFormat
        this.texture.type = THREE.FloatType
        this.texture.internalFormat = 'R32F'
        this.texture.minFilter = THREE.LinearFilter
        this.texture.magFilter = THREE.LinearFilter
        this.texture.generateMipmaps = false
        this.texture.needsUpdate = true
        this.texture.unpackAlignment = 1
        console.timeEnd('computeTexture') 
    }

    updateTexture()
    {
        this.texture.image.data.set(this.getTextureData())
        this.texture.needsUpdate = true
    }

    getTextureData()
    {
        return new Float32Array(this.tensor.dataSync())
    }

    dispose()
    {
        this.tensor?.dispose()
        this.texture?.dispose()
    }
}
