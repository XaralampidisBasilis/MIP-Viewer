let cachedContext = null

function isWindowsBrowser()
{
    const platform = navigator.userAgentData?.platform ?? navigator.platform ?? ''
    return /windows/i.test(platform)
}

function getAdapterOptions()
{
    if (isWindowsBrowser())
    {
        return {}
    }

    return { powerPreference: 'high-performance' }
}

function getRequiredLimits(adapter)
{
    const limits = {}

    if (adapter.limits.maxBufferSize)
    {
        limits.maxBufferSize = adapter.limits.maxBufferSize
    }

    if (adapter.limits.maxStorageBufferBindingSize)
    {
        limits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize
    }

    return limits
}

export async function getWebGPUComputeContext()
{
    if (cachedContext)
    {
        return cachedContext
    }

    if (!globalThis.navigator?.gpu)
    {
        throw new Error('WebGPU is not supported in this browser context.')
    }

    const adapter = await navigator.gpu.requestAdapter(getAdapterOptions())

    if (!adapter)
    {
        throw new Error('Unable to acquire a WebGPU adapter.')
    }

    const device = await adapter.requestDevice({
        requiredLimits: getRequiredLimits(adapter),
    })

    device.lost.then((info) =>
    {
        console.warn(`WebGPU device lost: ${info.reason}`, info.message)
        cachedContext = null
    })

    cachedContext = { adapter, device }
    return cachedContext
}

export async function createConfiguredWebGPUCanvas(canvas)
{
    const { adapter, device } = await getWebGPUComputeContext()
    const context = canvas.getContext('webgpu')

    if (!context)
    {
        throw new Error('Unable to acquire a WebGPU canvas context.')
    }

    const format = navigator.gpu.getPreferredCanvasFormat()

    context.configure({
        device,
        format,
        alphaMode: 'opaque',
    })

    return { adapter, device, context, format }
}
