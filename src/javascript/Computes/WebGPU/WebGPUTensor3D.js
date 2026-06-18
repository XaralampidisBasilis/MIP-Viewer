import { createBufferFromTypedArray, createStorageBuffer, readBuffer } from '../../WebGPU/WebGPUBufferUtils'

const DTYPE_INFO =
{
    float32: { ArrayType: Float32Array, bytesPerElement: 4 },
    uint32: { ArrayType: Uint32Array, bytesPerElement: 4 },
    int32: { ArrayType: Int32Array, bytesPerElement: 4 },
}

export class WebGPUTensor3D
{
    constructor(device, shape, dtype, buffer, label = 'tensor3d')
    {
        if (shape.length !== 3)
        {
            throw new Error(`WebGPUTensor3D expected rank-3 shape, got [${shape.join(', ')}].`)
        }

        if (!DTYPE_INFO[dtype])
        {
            throw new Error(`Unsupported WebGPU tensor dtype "${dtype}".`)
        }

        this.device = device
        this.shape = shape
        this.width = shape[0]
        this.height = shape[1]
        this.depth = shape[2]
        this.dtype = dtype
        this.buffer = buffer
        this.label = label
        this.size = shape[0] * shape[1] * shape[2]
        this.byteLength = this.size * DTYPE_INFO[dtype].bytesPerElement
    }

    static empty(device, shape, dtype = 'float32', label = 'empty-tensor3d')
    {
        const size = shape[0] * shape[1] * shape[2]
        const bytes = size * DTYPE_INFO[dtype].bytesPerElement
        const buffer = createStorageBuffer(device, bytes, label)

        return new WebGPUTensor3D(device, shape, dtype, buffer, label)
    }

    static fromTypedArray(device, shape, data, dtype = inferDtype(data), label = 'tensor3d')
    {
        const expectedSize = shape[0] * shape[1] * shape[2]

        if (data.length !== expectedSize)
        {
            throw new Error(`${label} expected ${expectedSize} values for shape [${shape.join(', ')}], got ${data.length}.`)
        }

        const buffer = createBufferFromTypedArray(device, data, GPUBufferUsage.STORAGE, label)

        return new WebGPUTensor3D(device, shape, dtype, buffer, label)
    }

    async read()
    {
        const { ArrayType } = DTYPE_INFO[this.dtype]
        return readBuffer(this.device, this.buffer, this.byteLength, ArrayType)
    }

    clone(label = `${this.label}:clone`)
    {
        const clone = WebGPUTensor3D.empty(this.device, this.shape, this.dtype, label)

        const encoder = this.device.createCommandEncoder()
        encoder.copyBufferToBuffer(this.buffer, 0, clone.buffer, 0, this.byteLength)
        this.device.queue.submit([encoder.finish()])

        return clone
    }

    dispose()
    {
        this.buffer?.destroy()
        this.buffer = null
    }
}

function inferDtype(data)
{
    if (data instanceof Float32Array) return 'float32'
    if (data instanceof Uint32Array) return 'uint32'
    if (data instanceof Int32Array) return 'int32'

    throw new Error(`Cannot infer WebGPU tensor dtype from ${data.constructor?.name ?? typeof data}.`)
}
