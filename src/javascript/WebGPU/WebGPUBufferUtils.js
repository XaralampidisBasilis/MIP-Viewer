export function alignTo(value, alignment)
{
    return Math.ceil(value / alignment) * alignment
}

function assertBufferSize(device, byteLength, label)
{
    const maxBufferSize = device.limits?.maxBufferSize
    const alignedByteLength = alignTo(byteLength, 4)

    if (maxBufferSize && alignedByteLength > maxBufferSize)
    {
        throw new Error(`${label} requires ${alignedByteLength} bytes, but this WebGPU device allows buffers up to ${maxBufferSize} bytes.`)
    }
}

export function createStorageBuffer(device, byteLength, label = 'storage-buffer')
{
    assertBufferSize(device, byteLength, label)

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
    const bytes = data.byteLength
    assertBufferSize(device, bytes, label)

    const buffer = device.createBuffer({
        label,
        size: alignTo(bytes, 16),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, bytes)
    return buffer
}

export function createBufferFromTypedArray(device, data, usage, label = 'gpu-buffer')
{
    assertBufferSize(device, data.byteLength, label)

    const buffer = device.createBuffer({
        label,
        size: alignTo(data.byteLength, 4),
        usage: usage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    })

    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength)
    return buffer
}

export async function readBuffer(device, source, byteLength, TypedArrayConstructor)
{
    assertBufferSize(device, byteLength, 'readback-buffer')

    const readback = device.createBuffer({
        label: 'readback-buffer',
        size: alignTo(byteLength, 4),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })

    const encoder = device.createCommandEncoder()
    encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength)
    device.queue.submit([encoder.finish()])

    await readback.mapAsync(GPUMapMode.READ)

    const mapped = readback.getMappedRange()
    const copy = mapped.slice(0, byteLength)
    readback.unmap()
    readback.destroy()

    return new TypedArrayConstructor(copy)
}
