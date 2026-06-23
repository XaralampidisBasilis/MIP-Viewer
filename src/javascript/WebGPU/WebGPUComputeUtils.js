

// Cache compute pipelines per GPU device.
// Creating pipelines can be expensive, so reusing them avoids unnecessary
// shader module and pipeline creation.
const pipelineCacheByDevice = new WeakMap()

function ceilDiv(value, divisor)
{
    // Divide value by divisor and round up.
    // This is useful for calculating how many compute workgroups are needed
    // when the data size is not evenly divisible by the workgroup size.
    return Math.ceil(value / divisor)
}

function getPipelineCache(device)
{
    // Get the pipeline cache associated with this specific WebGPU device.
    let cache = pipelineCacheByDevice.get(device)

    // Create a new cache for the device the first time it is used.
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

    // Use the shader code and entry point as the cache key.
    // If both are the same, the same compute pipeline can be reused.
    const key = `${entryPoint}\n${code}`
    const cached = cache.get(key)

    if (cached)
    {
        return cached
    }

    // Compile the WGSL shader code into a shader module.
    const module = device.createShaderModule({ label: `${label}:module`, code })

    // Create a compute pipeline from the shader module.
    // layout: 'auto' lets WebGPU infer the bind group layout from the shader.
    const pipeline = device.createComputePipeline({
        label: `${label}:pipeline`,
        layout: 'auto',
        compute: { module, entryPoint },
    })

    // Store the pipeline so future calls with the same code/entry point reuse it.
    cache.set(key, pipeline)
    return pipeline
}

function createBindGroup(device, label, pipeline, bindings)
{
    // Create a bind group for bind group index 0.
    // Each item in `bindings` is assigned to binding 0, 1, 2, ...
    // This helper currently assumes every binding is a GPU buffer.
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
    // Get a cached pipeline or create a new one for this shader.
    const pipeline = getComputePipeline(device, label, code, entryPoint)

    // Bind the provided GPU buffers to the shader.
    const bindGroup = createBindGroup(device, label, pipeline, bindings)

    // Record GPU commands into a command encoder.
    const encoder = device.createCommandEncoder({ label: `${label}:encoder` })

    // Begin a compute pass.
    const pass = encoder.beginComputePass({ label: `${label}:pass` })

    // Select the compute pipeline and bind group.
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)

    // Dispatch compute workgroups.
    // Missing Y or Z dispatch values default to 1.
    pass.dispatchWorkgroups(dispatch[0], dispatch[1] ?? 1, dispatch[2] ?? 1)

    // End the compute pass and submit the recorded commands to the GPU queue.
    pass.end()
    device.queue.submit([encoder.finish()])

    // Optionally wait until the GPU has finished the submitted work.
    // For best performance, avoid this unless a later CPU operation depends on it.
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
    // Nothing to run if no sequence steps were provided.
    if (steps.length === 0)
    {
        return
    }

    // Reuse one pipeline for all steps in the sequence.
    // This assumes every step uses the same shader code and entry point.
    const pipeline = getComputePipeline(device, label, code, entryPoint)

    // Record all compute dispatches into one command encoder and compute pass.
    const encoder = device.createCommandEncoder({ label: `${label}:encoder` })
    const pass = encoder.beginComputePass({ label: `${label}:pass` })

    pass.setPipeline(pipeline)

    for (let i = 0; i < steps.length; i += 1)
    {
        const step = steps[i]

        // Create a bind group for this step's buffers.
        // This allows each step to use different input/output buffers while
        // sharing the same compute pipeline.
        const bindGroup = createBindGroup(device, `${label}:${i}`, pipeline, step.bindings)

        pass.setBindGroup(0, bindGroup)

        // Dispatch this step's compute workgroups.
        // Missing Y or Z dispatch values default to 1.
        pass.dispatchWorkgroups(step.dispatch[0], step.dispatch[1] ?? 1, step.dispatch[2] ?? 1)
    }

    // End the pass and submit all sequence steps together.
    pass.end()
    device.queue.submit([encoder.finish()])

    // Wait for completion only when requested, or when resources need to be
    // destroyed safely after the GPU is done using them.
    if (awaitCompletion || disposeAfterSubmit.length > 0)
    {
        const done = device.queue.onSubmittedWorkDone()

        // Destroy temporary GPU resources after the submitted work completes.
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

        // Optionally await completion before returning to the caller.
        if (awaitCompletion)
        {
            await done
        }
    }
}

export function dispatchForShape(shape, workgroupSize)
{
    // This helper assumes the first three shape entries are:
    // [width, height, depth].
    const [width, height, depth, ] = shape

    // Calculate how many workgroups are needed in each dimension.
    // Values are rounded up so partially filled edge workgroups are included.
    return [
        ceilDiv(width, workgroupSize[0]),
        ceilDiv(height, workgroupSize[1]),
        ceilDiv(depth, workgroupSize[2]),
    ]
}

