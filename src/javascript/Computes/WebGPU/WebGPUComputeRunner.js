export function ceilDiv(value, divisor)
{
    return Math.ceil(value / divisor)
}

export async function runComputeProgram(
    device,
    {
        label = 'compute-program',
        code,
        entryPoint = 'main',
        bindings,
        dispatch,
    },
)
{
    const module = device.createShaderModule({ label: `${label}:module`, code })

    const pipeline = device.createComputePipeline({
        label: `${label}:pipeline`,
        layout: 'auto',
        compute: { module, entryPoint },
    })

    const bindGroup = device.createBindGroup({
        label: `${label}:bind-group`,
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((binding, index) => ({
            binding: index,
            resource: { buffer: binding.buffer },
        })),
    })

    const encoder = device.createCommandEncoder({ label: `${label}:encoder` })
    const pass = encoder.beginComputePass({ label: `${label}:pass` })

    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(dispatch[0], dispatch[1] ?? 1, dispatch[2] ?? 1)
    pass.end()

    device.queue.submit([encoder.finish()])
    await device.queue.onSubmittedWorkDone()
}

export function dispatchForShape(shape, workgroupSize = [8, 8, 4])
{
    const [depth, height, width] = shape

    return [
        ceilDiv(width, workgroupSize[0]),
        ceilDiv(height, workgroupSize[1]),
        ceilDiv(depth, workgroupSize[2]),
    ]
}
