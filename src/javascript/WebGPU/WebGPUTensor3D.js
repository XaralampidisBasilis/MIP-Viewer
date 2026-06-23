import { createBufferFromTypedArray, createStorageBuffer, readBuffer } from './WebGPUBufferUtils'

// Metadata for each supported tensor data type.
// Each dtype maps to the matching JavaScript typed array constructor
// and the number of bytes used by each element.
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
        // This class represents a scalar 3D tensor:
        // shape is always [width, height, depth].
        // Channels are intentionally not supported here.
        if (shape.length !== 3)
        {
            throw new Error(`WebGPUTensor3D expected shape [width, height, depth], got [${shape.join(', ')}].`)
        }

        // Make sure the requested dtype is one of the supported GPU data types.
        if (!DTYPE_INFO[dtype])
        {
            throw new Error(`Unsupported WebGPU tensor dtype "${dtype}".`)
        }

        // Store the WebGPU device used to create and operate on this tensor.
        this.device = device

        // Store dimensions individually for easier shader dispatch and indexing.
        this.width = shape[0]
        this.height = shape[1]
        this.depth = shape[2]

        // Store the normalized shape.
        this.shape = [this.width, this.height, this.depth]

        // Store the tensor's numeric type and backing GPU buffer.
        this.dtype = dtype
        this.buffer = buffer
        this.label = label

        // Total number of scalar values in the tensor.
        this.size = this.width * this.height * this.depth

        // Total number of bytes required by the tensor data.
        this.byteLength = this.size * DTYPE_INFO[dtype].bytesPerElement
    }

    static empty(device, shape, dtype = 'float32', label = 'empty-tensor3d')
    {
        validateShape(shape)

        // Compute the total number of scalar values.
        const size = shape[0] * shape[1] * shape[2]

        // Convert element count into byte count based on the dtype.
        const bytes = size * DTYPE_INFO[dtype].bytesPerElement

        // Allocate an empty GPU storage buffer large enough for the tensor.
        const buffer = createStorageBuffer(device, bytes, label)

        // Wrap the GPU buffer in a WebGPUTensor3D instance.
        return new WebGPUTensor3D(device, shape, dtype, buffer, label)
    }

    static fromTypedArray(device, shape, data, dtype = inferDtype(data), label = 'tensor3d')
    {
        validateShape(shape)

        // A tensor with shape [width, height, depth] must contain exactly
        // width * height * depth scalar values.
        const expectedSize = shape[0] * shape[1] * shape[2]

        // Prevent uploading incorrectly sized CPU data into the GPU tensor.
        if (data.length !== expectedSize)
        {
            throw new Error(`${label} expected ${expectedSize} values for shape [${shape.join(', ')}], got ${data.length}.`)
        }

        // Create a GPU storage buffer and upload the typed array into it.
        const buffer = createBufferFromTypedArray(device, data, GPUBufferUsage.STORAGE, label)

        // Wrap the uploaded GPU buffer in a WebGPUTensor3D instance.
        return new WebGPUTensor3D(device, shape, dtype, buffer, label)
    }

    async read()
    {
        // Pick the correct typed array constructor for this tensor's dtype.
        const { ArrayType } = DTYPE_INFO[this.dtype]

        // Copy the GPU buffer back to CPU memory and return it as a typed array.
        return readBuffer(this.device, this.buffer, this.byteLength, ArrayType)
    }

    clone(label = `${this.label}:clone`)
    {
        // Allocate a new GPU tensor with the same shape and dtype.
        const clone = WebGPUTensor3D.empty(this.device, this.shape, this.dtype, label)

        // Copy the current tensor's GPU buffer into the clone's GPU buffer.
        const encoder = this.device.createCommandEncoder()
        encoder.copyBufferToBuffer(this.buffer, 0, clone.buffer, 0, this.byteLength)
        this.device.queue.submit([encoder.finish()])

        return clone
    }

    dispose()
    {
        // Release the GPU buffer when this tensor is no longer needed.
        this.buffer?.destroy()

        // Clear the reference to avoid accidentally using a destroyed buffer.
        this.buffer = null
    }
}

function validateShape(shape)
{
    if (shape.length !== 3)
    {
        throw new Error(`Expected tensor shape [width, height, depth], got [${shape.join(', ')}].`)
    }

    const [width, height, depth] = shape

    if (!Number.isInteger(width) || !Number.isInteger(height) || !Number.isInteger(depth))
    {
        throw new Error(`Tensor shape values must be integers, got [${shape.join(', ')}].`)
    }

    if (width <= 0 || height <= 0 || depth <= 0)
    {
        throw new Error(`Tensor shape values must be positive, got [${shape.join(', ')}].`)
    }
}

function inferDtype(data)
{
    // Infer the tensor dtype from the JavaScript typed array class.
    if (data instanceof Float32Array) return 'float32'
    if (data instanceof Uint32Array) return 'uint32'
    if (data instanceof Int32Array) return 'int32'

    // Reject unsupported input arrays, such as Uint8Array or normal JS arrays.
    throw new Error(`Cannot infer WebGPU tensor dtype from ${data.constructor?.name ?? typeof data}.`)
}