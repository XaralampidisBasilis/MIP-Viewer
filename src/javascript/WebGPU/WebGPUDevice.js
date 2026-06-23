// Cache the WebGPU adapter/device pair so the app does not request a new device
// every time compute or canvas setup is needed.
let cachedContext = null

function isWindowsBrowser()
{
    // Prefer the modern userAgentData API when available, then fall back to
    // navigator.platform for older browsers.
    const platform = navigator.userAgentData?.platform ?? navigator.platform ?? ''
    return /windows/i.test(platform)
}

function getAdapterOptions()
{
    // On Windows, avoid forcing high-performance mode. Some Windows/browser/GPU
    // combinations behave better when the browser chooses the adapter.
    if (isWindowsBrowser())
    {
        return {}
    }

    // On other platforms, request the higher-performance GPU when available.
    return { powerPreference: 'high-performance' }
}

function getRequiredLimits(adapter)
{
    const limits = {}

    // Request the adapter's maximum buffer size when the limit is exposed.
    // This allows larger GPU buffers for compute workloads.
    if (adapter.limits.maxBufferSize)
    {
        limits.maxBufferSize = adapter.limits.maxBufferSize
    }

    // Request the adapter's maximum storage buffer binding size when available.
    // This is useful for compute shaders that bind large storage buffers.
    if (adapter.limits.maxStorageBufferBindingSize)
    {
        limits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize
    }

    return limits
}

function getRequiredFeatures(adapter)
{
    return [...adapter.features]
}

export async function getWebGPUComputeContext()
{
    // Reuse the existing adapter/device unless the device has been lost.
    if (cachedContext)
    {
        return cachedContext
    }

    // WebGPU is only available in supporting browsers and secure contexts.
    if (!globalThis.navigator?.gpu)
    {
        throw new Error('WebGPU is not supported in this browser context.')
    }

    // Ask the browser for a GPU adapter using platform-specific preferences.
    const adapter = await navigator.gpu.requestAdapter(getAdapterOptions())

    if (!adapter)
    {
        throw new Error('Unable to acquire a WebGPU adapter.')
    }

    // Create a GPU device with the largest useful limits exposed by the adapter.
    const device = await adapter.requestDevice({
        requiredFeatures: getRequiredFeatures(adapter),
        requiredLimits: getRequiredLimits(adapter),
    })

    // If the device is lost, clear the cache so the next call can recreate
    // a fresh WebGPU context.
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
    // Reuse the shared adapter/device used by compute code.
    const { adapter, device } = await getWebGPUComputeContext()

    // Get the WebGPU rendering context from the provided canvas element.
    const context = canvas.getContext('webgpu')

    if (!context)
    {
        throw new Error('Unable to acquire a WebGPU canvas context.')
    }

    // Use the browser's preferred canvas texture format for best compatibility.
    const format = navigator.gpu.getPreferredCanvasFormat()

    // Configure the canvas so WebGPU can present rendered frames to it.
    context.configure({
        device,
        format,
        alphaMode: 'opaque',
    })

    return { adapter, device, context, format }
}

