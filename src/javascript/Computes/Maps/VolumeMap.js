import * as THREE from 'three'
import { Storage3DTexture } from 'three/webgpu'
import Computes from '../Computes'
import { toHalfFloat } from '../../Utils/DataUtils'
import { getWebGPUComputeContext } from '../../WebGPU/WebGPUDevice'
import { WebGPUTensor3D } from '../../WebGPU/WebGPUTensor3D'
import { map3dInPlaceWebGPU, reduceMinMaxWebGPU, resizeTrilinearWebGPU } from '../WebGPU/WebGPUVolumeKernels'

const VOLUME_TEXTURE_WORKGROUP_SIZE = [8, 8, 4]

export default class VolumeMap
{
    constructor()
    {
        this.computes = new Computes()
        this.configs = this.computes.configs
        this.resources = this.computes.resources
        this.renderer = this.computes.renderer
        this.webgpuTensor = null
    }

    setVolume()
    {
        this.volume = this.resources.items.volume
        this.shape = this.volume.dimensions.toReversed()
        this.dimensions = new THREE.Vector3().fromArray(this.volume.dimensions)
        this.spacing = new THREE.Vector3().fromArray(this.volume.spacing)
        this.size = new THREE.Vector3().fromArray(this.volume.size)
    }

    async computeTensor()
    {
        await this.computeTensorWebGPU()
    }

    async computeTensorWebGPU()
    {
        console.time('computeTensor@WebGPU') 
        
        this.setVolume()
        this.tensor?.dispose()
        this.tensor = null
        this.webgpuTensor?.dispose()
        this.webgpuTensor = null
        this.texture?.dispose()
        this.texture = null
        this.textureData = null

        const { device } = await getWebGPUComputeContext()
        const shape = this.volume.dimensions
        const data = new Float32Array(this.volume.data)
        const raw = WebGPUTensor3D.fromTypedArray(device, shape, data, 'float32', 'volume-raw')
        const [minValue, maxValue] = await reduceMinMaxWebGPU(raw)

        this.minValue = minValue
        this.maxValue = maxValue

        let tensor = await map3dInPlaceWebGPU(raw, this.minValue, this.maxValue)

        if (this.configs.downscaleEnabled)
        {
            const spacing = this.volume.spacing
            const newShape = shape.map((x) => Math.ceil(this.configs.downscaleFactor * x))
            const newSpacing = spacing.map((x, i) => shape[i] / newShape[i] * x)
            const resized = await resizeTrilinearWebGPU(tensor, newShape, false, true)

            tensor.dispose()
            tensor = resized

            this.shape = newShape
            this.dimensions.fromArray(newShape)
            this.spacing.fromArray(newSpacing)
        }
        else
        {
            this.shape = shape
        }

        this.webgpuTensor = tensor

        console.log(this)
        console.timeEnd('computeTensor@WebGPU') 
    }

    async computeTexture()
    {
        console.time('computeTexture@WebGPU')

        if (!this.webgpuTensor)
        {
            throw new Error('VolumeMap.computeTexture expected computeTensorWebGPU() to run first.')
        }

        this.texture?.dispose()
        this.textureData = null

        if (this.canUseFloat32StorageTexture())
        {
            this.texture = await this.createFloat32StorageTexture()
            console.timeEnd('computeTexture@WebGPU')
            return
        }

        const data = await this.webgpuTensor.read()
        this.textureData = this.float32ToHalfFloatData(data)
        this.texture = new THREE.Data3DTexture(this.textureData, ...this.dimensions)
        this.texture.format = THREE.RedFormat
        this.texture.type = THREE.HalfFloatType
        this.texture.minFilter = THREE.LinearFilter
        this.texture.magFilter = THREE.LinearFilter
        this.texture.generateMipmaps = false
        this.texture.needsUpdate = true
        this.texture.unpackAlignment = 2
        console.timeEnd('computeTexture@WebGPU')
    }

    updateTexture()
    {
        throw new Error('VolumeMap.updateTexture is not supported for the WebGPU-only volume path.')
    }

    computeTextureData()
    {
        throw new Error('VolumeMap.computeTextureData is not supported for the WebGPU-only volume path.')
    }

    getTextureData()
    {
        throw new Error('VolumeMap.getTextureData is not supported for the WebGPU-only volume path.')
    }

    canUseFloat32StorageTexture()
    {
        const renderer = this.renderer?.instance
        const device = this.webgpuTensor?.device

        return Boolean(
            renderer?.isWebGPURenderer &&
            this.renderer?.ready &&
            renderer.backend?.device === device &&
            device?.features?.has?.('float32-filterable')
        )
    }

    async createFloat32StorageTexture()
    {
        const renderer = this.renderer.instance
        const device = this.webgpuTensor.device
        const [width, height, depth] = this.shape
        const texture = new Storage3DTexture(width, height, depth)

        texture.name = 'volume-map-float32'
        texture.format = THREE.RedFormat
        texture.type = THREE.FloatType
        texture.minFilter = THREE.LinearFilter
        texture.magFilter = THREE.LinearFilter
        texture.generateMipmaps = false
        texture.mipmapsAutoUpdate = false
        texture.needsUpdate = true

        renderer.initTexture(texture)

        const textureGPU = renderer.backend.get(texture).texture
        const encoder = device.createCommandEncoder({ label: 'volume-buffer-to-texture:encoder' })
        const pass = encoder.beginComputePass({ label: 'volume-buffer-to-texture:pass' })
        const pipeline = device.createComputePipeline({
            label: 'volume-buffer-to-texture:pipeline',
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    label: 'volume-buffer-to-texture:module',
                    code: volumeBufferToTextureWGSL(this.shape),
                }),
                entryPoint: 'main',
            },
        })
        const bindGroup = device.createBindGroup({
            label: 'volume-buffer-to-texture:bind-group',
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.webgpuTensor.buffer } },
                { binding: 1, resource: textureGPU.createView({ dimension: '3d' }) },
            ],
        })
        const dispatch = dispatchForShape(this.shape, VOLUME_TEXTURE_WORKGROUP_SIZE)

        pass.setPipeline(pipeline)
        pass.setBindGroup(0, bindGroup)
        pass.dispatchWorkgroups(dispatch[0], dispatch[1], dispatch[2])
        pass.end()

        device.queue.submit([encoder.finish()])
        await device.queue.onSubmittedWorkDone()

        texture.userData.volumeTextureBackend = 'webgpu-storage-float32'
        return texture
    }

    float32ToHalfFloatData(data)
    {
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
        this.webgpuTensor?.dispose()
        this.texture?.dispose()
    }
}

function dispatchForShape(shape, workgroupSize)
{
    return [
        Math.ceil(shape[0] / workgroupSize[0]),
        Math.ceil(shape[1] / workgroupSize[1]),
        Math.ceil(shape[2] / workgroupSize[2]),
    ]
}

function volumeBufferToTextureWGSL(shape)
{
    const [width, height, depth] = shape

    return /* wgsl */ `
const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const DEPTH: u32 = ${depth}u;

@group(0) @binding(0) var<storage, read> volume: array<f32>;
@group(0) @binding(1) var volume_texture: texture_storage_3d<r32float, write>;

fn index3(gid: vec3<u32>) -> u32
{
    return gid.z * HEIGHT * WIDTH + gid.y * WIDTH + gid.x;
}

@compute @workgroup_size(${VOLUME_TEXTURE_WORKGROUP_SIZE[0]}, ${VOLUME_TEXTURE_WORKGROUP_SIZE[1]}, ${VOLUME_TEXTURE_WORKGROUP_SIZE[2]})
fn main(@builtin(global_invocation_id) gid: vec3<u32>)
{
    if (gid.x >= WIDTH || gid.y >= HEIGHT || gid.z >= DEPTH) {
        return;
    }

    let value = volume[index3(gid)];
    textureStore(volume_texture, vec3<i32>(gid), vec4<f32>(value, 0.0, 0.0, 1.0));
}
`
}
