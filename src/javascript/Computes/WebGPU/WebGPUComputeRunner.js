export function ceilDiv(value, divisor)
{
    return Math.ceil(value / divisor)
}

const pipelineCacheByDevice = new WeakMap()

function getPipelineCache(device)
{
    let cache = pipelineCacheByDevice.get(device)

    if (!cache)
    {
        cache = new Map()
        pipelineCacheByDevice.set(device, cache)
    }

    return cache
}

function getComputePipeline(device, label, code, entryPoint)
{
    const cache = getPipelineCache(device)
    const key = `${entryPoint}\n${code}`
    const cached = cache.get(key)

    if (cached)
    {
        return cached
    }

    const module = device.createShaderModule({ label: `${label}:module`, code })
    const pipeline = device.createComputePipeline({
        label: `${label}:pipeline`,
        layout: 'auto',
        compute: { module, entryPoint },
    })

    cache.set(key, pipeline)
    return pipeline
}

function createBindGroup(device, label, pipeline, bindings)
{
    return device.createBindGroup({
        label: `${label}:bind-group`,
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((binding, index) => ({
            binding: index,
            resource: { buffer: binding.buffer },
        })),
    })
}

export async function runComputeProgram(
    device,
    {
        label = 'compute-program',
        code,
        entryPoint = 'main',
        bindings,
        dispatch,
        awaitCompletion = false,
    },
)
{
    const pipeline = getComputePipeline(device, label, code, entryPoint)
    const bindGroup = createBindGroup(device, label, pipeline, bindings)

    const encoder = device.createCommandEncoder({ label: `${label}:encoder` })
    const pass = encoder.beginComputePass({ label: `${label}:pass` })

    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(dispatch[0], dispatch[1] ?? 1, dispatch[2] ?? 1)
    pass.end()

    device.queue.submit([encoder.finish()])

    if (awaitCompletion)
    {
        await device.queue.onSubmittedWorkDone()
    }
}

export async function runComputeProgramSequence(
    device,
    {
        label = 'compute-program-sequence',
        code,
        entryPoint = 'main',
        steps,
        awaitCompletion = false,
        disposeAfterSubmit = [],
    },
)
{
    if (steps.length === 0)
    {
        return
    }

    const pipeline = getComputePipeline(device, label, code, entryPoint)
    const encoder = device.createCommandEncoder({ label: `${label}:encoder` })
    const pass = encoder.beginComputePass({ label: `${label}:pass` })

    pass.setPipeline(pipeline)

    for (let i = 0; i < steps.length; i += 1)
    {
        const step = steps[i]
        const bindGroup = createBindGroup(device, `${label}:${i}`, pipeline, step.bindings)

        pass.setBindGroup(0, bindGroup)
        pass.dispatchWorkgroups(step.dispatch[0], step.dispatch[1] ?? 1, step.dispatch[2] ?? 1)
    }

    pass.end()
    device.queue.submit([encoder.finish()])

    if (awaitCompletion || disposeAfterSubmit.length > 0)
    {
        const done = device.queue.onSubmittedWorkDone()

        if (disposeAfterSubmit.length > 0)
        {
            done.then(() =>
            {
                for (const resource of disposeAfterSubmit)
                {
                    resource?.destroy?.()
                }
            })
        }

        if (awaitCompletion)
        {
            await done
        }
    }
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
