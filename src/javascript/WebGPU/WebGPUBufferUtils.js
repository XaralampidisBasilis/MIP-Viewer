
export function alignTo(value, alignment)
{
    // Round value up to the nearest multiple of alignment.
    // WebGPU buffer sizes often need to be aligned to 4 or 16 bytes.
    return Math.ceil(value / alignment) * alignment
}

function assertBufferSize(device, byteLength, label)
{
    // Read the maximum allowed buffer size for this WebGPU device, if exposed.
    const maxBufferSize = device.limits?.maxBufferSize

    // WebGPU buffer sizes must be aligned to at least 4 bytes.
    const alignedByteLength = alignTo(byteLength, 4)

    // Fail early with a clear message if the requested buffer would exceed
    // the device's supported maximum.
    if (maxBufferSize && alignedByteLength > maxBufferSize)
    {
        throw new Error(`${label} requires ${alignedByteLength} bytes, but this WebGPU device allows buffers up to ${maxBufferSize} bytes.`)
    }
}

export function createStorageBuffer(device, byteLength, label = 'storage-buffer')
{
    // Validate the requested buffer size before attempting GPU allocation.
    assertBufferSize(device, byteLength, label)

    // Create a storage buffer that can be used by shaders and copied to/from.
    return device.createBuffer({
        label,
        size: alignTo(byteLength, 4),
        usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
    })
}

export function createUniformBuffer(device, data, label = 'uniform-buffer')
{
    // Use the exact byte length of the supplied typed array or DataView.
    const bytes = data.byteLength

    // Make sure the uniform data will fit within the device limits.
    assertBufferSize(device, bytes, label)

    // Uniform buffers commonly require 16-byte alignment.
    const buffer = device.createBuffer({
        label,
        size: alignTo(bytes, 16),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Upload the provided CPU-side data into the GPU buffer.
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, bytes)
    return buffer
}

export function createBufferFromTypedArray(device, data, usage, label = 'gpu-buffer')
{
    // Confirm the typed array can fit in a GPU buffer on this device.
    assertBufferSize(device, data.byteLength, label)

    // Create a general-purpose GPU buffer with the caller's requested usage,
    // plus copy flags so it can be uploaded to or read back from later.
    const buffer = device.createBuffer({
        label,
        size: alignTo(data.byteLength, 4),
        usage: usage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    })

    // Upload the typed array contents into the new GPU buffer.
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength)
    return buffer
}

export async function readBuffer(device, source, byteLength, TypedArrayConstructor)
{
    // Validate the size of the temporary readback buffer.
    assertBufferSize(device, byteLength, 'readback-buffer')

    // Create a CPU-readable buffer used as the copy destination.
    const readback = device.createBuffer({
        label: 'readback-buffer',
        size: alignTo(byteLength, 4),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })

    // Copy the requested bytes from the GPU source buffer into the readback buffer.
    const encoder = device.createCommandEncoder()
    encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength)
    device.queue.submit([encoder.finish()])

    // Wait until the GPU data is available for CPU-side reading.
    await readback.mapAsync(GPUMapMode.READ)

    // Copy the mapped data before unmapping, because the mapped range becomes
    // invalid once the buffer is unmapped.
    const mapped = readback.getMappedRange()
    const copy = mapped.slice(0, byteLength)

    // Release GPU mapping resources and destroy the temporary buffer.
    readback.unmap()
    readback.destroy()

    // Return the copied bytes as the caller's requested typed array type.
    return new TypedArrayConstructor(copy)
}

