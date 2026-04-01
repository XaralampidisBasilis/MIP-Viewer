import * as tf from '@tensorflow/tfjs'

export type GLMemoryPolicy = 
{
    deleteTextureThreshold: number
    flushThreshold: number
    lazilyUnpack: boolean
    pack: boolean
}

export type GLTexturePoolStats = 
{
    numUsedTextures: number
    numFreeTextures: number
    numBytesAllocated: number
    numBytesFree: number
}

export type GLPurgeFreeTexturePoolResult = 
{
    deletedTextures: number
    freedBytes: number
    before: GLTexturePoolStats
    after: GLTexturePoolStats
}

export type GLMemorySnapshot = 
{
    numTensors: number
    numDataBuffers: number
    numBytes: number
    unreliable: boolean
    numBytesInGPU: number
    numUsedTextures: number
    numFreeTextures: number
    numBytesAllocated: number
    numBytesFree: number
    numBytesUsed: number
}

export type GLPeakTracker = 
{
    label: string
    samples: number
    peakNumTensors: number
    peakNumDataBuffers: number
    peakNumBytes: number
    peakNumBytesInGPU: number
    peakNumBytesAllocated: number
    peakNumBytesFree: number
    peakNumBytesUsed: number
    lastSnapshot: GLMemorySnapshot | null
}

export type GLPeakCounter = Exclude<keyof GLPeakTracker, 'label' | 'lastSnapshot'>

export type GLExecutionProfile = 
{
    label: string
    timeMs: number
    samples: number
    startSnapshot: GLMemorySnapshot
    endSnapshot: GLMemorySnapshot
    peakSnapshot: GLMemorySnapshot
}

export type GLExecutionProfileOptions = 
{
    label?: string
    flushBeforeStart?: boolean
    flushAfterEnd?: boolean
    finishAfterEnd?: boolean
    log?: boolean
    logLevel?: 'summary' | 'full'
    onProfile?: (profile: GLExecutionProfile) => void
}

export const GLMemoryProfiles = Object.freeze
({
    aggressive: 
    {
        deleteTextureThreshold: 0,
        flushThreshold: 1,
    },
    balanced:
    {
        deleteTextureThreshold: 0x10000000, // 256MB
        flushThreshold: 1,
    },
    performance:
    {
        deleteTextureThreshold: -1,
        flushThreshold: -1,
    },
})

function validateDeleteTextureThreshold(value: number): number
{
    // Validate delete-threshold input before applying it to tf.env().
    if (!Number.isFinite(value))
    {
        throw new Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be finite, got ${value}`)
    }

    if (value < 0 && value !== -1)
    {
        throw new Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be -1 or >= 0, got ${value}`)
    }

    return value
}

function validateFlushThreshold(value: number): number
{
    // Validate flush-threshold input before applying it to tf.env().
    if (!Number.isFinite(value))
    {
        throw new Error(`WEBGL_FLUSH_THRESHOLD must be finite, got ${value}`)
    }

    if (value < 0 && value !== -1)
    {
        throw new Error(`WEBGL_FLUSH_THRESHOLD must be -1 or >= 0, got ${value}`)
    }

    return value
}

function bytesPerElement(dtype: tf.DataType): number
{
    if (dtype === 'float32' || dtype === 'int32') return 4
    if (dtype === 'bool') return 1
    if (dtype === 'complex64') return 8
    if (dtype === 'string') return 0

    return 4
}

function getWebGLBackend(): any | null
{
    // Return TFJS WebGL backend instance, or null when another backend is active.
    if (tf.getBackend() !== 'webgl') return null

    return tf.backend()
}

function getWebGLContextFromBackend(backend: any): WebGLRenderingContext | WebGL2RenderingContext | null
{
    // Resolve raw WebGL context from backend internals (handles multiple TFJS shapes).
    const gpgpu = backend?.getGPGPUContext?.() ?? backend?.gpgpu
    const gl = gpgpu?.gl

    return gl ?? null
}

export function isWebGLBackend(): boolean
{
    // Use when logic should run only for TFJS WebGL backend.
    return tf.getBackend() === 'webgl'
}

export function getWebGLContext(): WebGLRenderingContext | WebGL2RenderingContext | null
{
    // Use when you need direct GL operations like flush() or finish().
    const backend = getWebGLBackend()
    if (!backend) return null

    return getWebGLContextFromBackend(backend)
}

export function getGLMemoryPolicy(): GLMemoryPolicy
{
    // Read current TFJS WebGL memory-related env settings.
    return {
        deleteTextureThreshold: tf.env().getNumber('WEBGL_DELETE_TEXTURE_THRESHOLD'),
        flushThreshold: tf.env().getNumber('WEBGL_FLUSH_THRESHOLD'),
        lazilyUnpack: tf.env().getBool('WEBGL_LAZILY_UNPACK'),
        pack: tf.env().getBool('WEBGL_PACK'),
    }
}

export function getTensorNumBytes(tensor: tf.Tensor): number
{
    // Estimate tensor byte size from dtype and element count.
    return tensor.size * bytesPerElement(tensor.dtype)
}

export function getMaxTensorNumBytes(tensors?: tf.Tensor | tf.Tensor[] | null): number
{
    // Return the largest estimated tensor size in bytes from one or many tensors.
    if (!tensors) return 0

    const list = Array.isArray(tensors) ? tensors : [tensors]
    let maxBytes = 0
    for (const tensor of list)
    {
        const bytes = getTensorNumBytes(tensor)
        if (bytes > maxBytes) maxBytes = bytes
    }

    return maxBytes
}

export function suggestDeleteTextureThresholdFromTensors(
    tensors?: tf.Tensor | tf.Tensor[] | null,
    multiplier: number = 1,
    floorBytes: number = 0
): number
{
    // Suggest threshold from tensor sizes so policy can adapt to workload shapes.
    if (!Number.isFinite(multiplier) || multiplier <= 0)
    {
        throw new Error(`multiplier must be > 0 and finite, got ${multiplier}`)
    }

    if (!Number.isFinite(floorBytes) || floorBytes < 0)
    {
        throw new Error(`floorBytes must be >= 0 and finite, got ${floorBytes}`)
    }

    const maxTensorBytes = getMaxTensorNumBytes(tensors)
    const threshold = Math.ceil(maxTensorBytes * multiplier)

    return Math.max(threshold, floorBytes)
}

export function setDeleteTextureThresholdFromTensors(
    tensors?: tf.Tensor | tf.Tensor[] | null,
    multiplier: number = 1,
    floorBytes: number = 0
): number
{
    // Compute and apply delete-threshold derived from tensor sizes.
    const threshold = suggestDeleteTextureThresholdFromTensors(tensors, multiplier, floorBytes)
    return setDeleteTextureThreshold(threshold)
}

export function setDeleteTextureThreshold(bytes: number): number
{
    // Set texture reuse/deletion threshold in bytes (-1 keeps textures for reuse).
    const value = validateDeleteTextureThreshold(bytes)
    tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', value)
    return value
}

export function setFlushThreshold(ms: number): number
{
    // Set auto-flush threshold in milliseconds (-1 disables auto time-based flushes).
    const value = validateFlushThreshold(ms)
    tf.env().set('WEBGL_FLUSH_THRESHOLD', value)
    return value
}

export function setGLMemoryPolicy(policy: Partial<GLMemoryPolicy>): GLMemoryPolicy
{
    // Apply multiple WebGL memory flags in one call and return the effective policy.
    if (policy.deleteTextureThreshold !== undefined)
    {
        setDeleteTextureThreshold(policy.deleteTextureThreshold)
    }

    if (policy.flushThreshold !== undefined)
    {
        setFlushThreshold(policy.flushThreshold)
    }

    if (policy.lazilyUnpack !== undefined)
    {
        tf.env().set('WEBGL_LAZILY_UNPACK', policy.lazilyUnpack)
    }

    if (policy.pack !== undefined)
    {
        tf.env().set('WEBGL_PACK', policy.pack)
    }

    return getGLMemoryPolicy()
}

export async function withGLMemoryPolicy<T>(policy: Partial<GLMemoryPolicy>, fn: () => T | Promise<T>): Promise<T>
{
    // Temporarily apply a policy only around one workload, then restore previous values.
    const prev = getGLMemoryPolicy()

    setGLMemoryPolicy(policy)
    try
    {
        return await fn()
    }
    finally
    {
        setGLMemoryPolicy(prev)
    }
}

export function applyAggressiveGLMemoryPolicy(): GLMemoryPolicy
{
    // Prefer lower memory peaks; useful on low-end GPUs (may be slower).
    return setGLMemoryPolicy(GLMemoryProfiles.aggressive)
}

export function applyBalancedGLMemoryPolicy(): GLMemoryPolicy
{
    // Balanced preset between memory usage and performance.
    return setGLMemoryPolicy(GLMemoryProfiles.balanced)
}

export function applyPerformanceGLMemoryPolicy(): GLMemoryPolicy
{
    // Prefer speed and texture reuse; may increase peak GPU memory.
    return setGLMemoryPolicy(GLMemoryProfiles.performance)
}

export function flushWebGL(): boolean
{
    // Submit queued GL commands to the driver; non-blocking in most cases.
    const gl = getWebGLContext()
    if (!gl) return false

    gl.flush()
    return true
}

export function finishWebGL(): boolean
{
    // Force GPU completion of queued work; useful for strict sync/debug, often slower.
    const gl = getWebGLContext()
    if (!gl) return false

    gl.finish()
    return true
}

export async function disposeFlushYield(tensors?: tf.Tensor | tf.Tensor[] | null, useFinish: boolean = false): Promise<void>
{
    // Convenience helper for heavy loops: dispose tensors, flush GL, optionally finish, then yield.
    if (tensors)
    {
        tf.dispose(tensors)
    }

    flushWebGL()
    if (useFinish) finishWebGL()

    await tf.nextFrame()
}

export function getGLMemorySnapshot(): GLMemorySnapshot
{
    // Read combined tf.memory + WebGL texture pool counters in one snapshot.
    const mem = tf.memory() as ReturnType<typeof tf.memory> & { numBytesInGPU?: number }
    const stats = getGLTexturePoolStats()

    const numBytesAllocated = stats?.numBytesAllocated ?? 0
    const numBytesFree = stats?.numBytesFree ?? 0
    const numBytesUsed = Math.max(0, numBytesAllocated - numBytesFree)
    const numBytesInGPU = mem.numBytesInGPU ?? numBytesUsed

    return {
        numTensors: mem.numTensors,
        numDataBuffers: mem.numDataBuffers,
        numBytes: mem.numBytes,
        unreliable: mem.unreliable ?? false,
        numBytesInGPU,
        numUsedTextures: stats?.numUsedTextures ?? 0,
        numFreeTextures: stats?.numFreeTextures ?? 0,
        numBytesAllocated,
        numBytesFree,
        numBytesUsed,
    }
}

function isHigherSnapshot(current: GLMemorySnapshot, peak: GLMemorySnapshot): boolean
{
    return (
        current.numBytesAllocated > peak.numBytesAllocated ||
        current.numBytesInGPU > peak.numBytesInGPU ||
        current.numBytes > peak.numBytes ||
        current.numTensors > peak.numTensors ||
        current.numDataBuffers > peak.numDataBuffers
    )
}

function logGLExecutionProfile(profile: GLExecutionProfile, level: 'summary' | 'full' = 'summary'): void
{
    if (level === 'summary')
    {
        console.log(
            `[${profile.label}] timeMs=${profile.timeMs.toFixed(3)} samples=${profile.samples} peakAllocated=${profile.peakSnapshot.numBytesAllocated} peakInGPU=${profile.peakSnapshot.numBytesInGPU} peakTensors=${profile.peakSnapshot.numTensors}`
        )
        return
    }

    console.log(
        `[${profile.label}] timeMs=${profile.timeMs.toFixed(3)} samples=${profile.samples}`
    )
    logGLMemorySnapshot(`${profile.label}:start`, profile.startSnapshot)
    logGLMemorySnapshot(`${profile.label}:end`, profile.endSnapshot)
    logGLMemorySnapshot(`${profile.label}:peak`, profile.peakSnapshot)
}

export function profileGLExecution<T>(
    fn: (samplePeak: () => GLMemorySnapshot) => T,
    options: GLExecutionProfileOptions = {}
): T
{
    // Profile a sync GL segment and return callback result.
    const label = options.label ?? 'glProfile'
    const flushBeforeStart = options.flushBeforeStart ?? false
    const flushAfterEnd = options.flushAfterEnd ?? false
    const finishAfterEnd = options.finishAfterEnd ?? false
    const shouldLog = options.log ?? true
    const logLevel = options.logLevel ?? 'summary'

    if (flushBeforeStart) flushWebGL()

    const startSnapshot = getGLMemorySnapshot()
    let peakSnapshot = startSnapshot
    let samples = 1

    const samplePeak = (): GLMemorySnapshot =>
    {
        const snapshot = getGLMemorySnapshot()
        samples += 1

        if (isHigherSnapshot(snapshot, peakSnapshot))
        {
            peakSnapshot = snapshot
        }

        return snapshot
    }

    const t0 = performance.now()
    const result = fn(samplePeak)
    samplePeak()

    if (flushAfterEnd) flushWebGL()
    if (finishAfterEnd) finishWebGL()

    const endSnapshot = getGLMemorySnapshot()
    samples += 1
    if (isHigherSnapshot(endSnapshot, peakSnapshot))
    {
        peakSnapshot = endSnapshot
    }

    const profile: GLExecutionProfile = {
        label,
        timeMs: performance.now() - t0,
        samples,
        startSnapshot,
        endSnapshot,
        peakSnapshot,
    }

    options.onProfile?.(profile)
    if (shouldLog) logGLExecutionProfile(profile, logLevel)

    return result
}

export async function profileGLExecutionAsync<T>(
    fn: (samplePeak: () => GLMemorySnapshot) => T | Promise<T>,
    options: GLExecutionProfileOptions = {}
): Promise<T>
{
    // Async variant of profileGLExecution for callbacks that await.
    const label = options.label ?? 'glProfile'
    const flushBeforeStart = options.flushBeforeStart ?? false
    const flushAfterEnd = options.flushAfterEnd ?? false
    const finishAfterEnd = options.finishAfterEnd ?? false
    const shouldLog = options.log ?? true
    const logLevel = options.logLevel ?? 'summary'

    if (flushBeforeStart) flushWebGL()

    const startSnapshot = getGLMemorySnapshot()
    let peakSnapshot = startSnapshot
    let samples = 1

    const samplePeak = (): GLMemorySnapshot =>
    {
        const snapshot = getGLMemorySnapshot()
        samples += 1

        if (isHigherSnapshot(snapshot, peakSnapshot))
        {
            peakSnapshot = snapshot
        }

        return snapshot
    }

    const t0 = performance.now()
    const result = await fn(samplePeak)
    samplePeak()

    if (flushAfterEnd) flushWebGL()
    if (finishAfterEnd) finishWebGL()

    const endSnapshot = getGLMemorySnapshot()
    samples += 1
    if (isHigherSnapshot(endSnapshot, peakSnapshot))
    {
        peakSnapshot = endSnapshot
    }

    const profile: GLExecutionProfile = {
        label,
        timeMs: performance.now() - t0,
        samples,
        startSnapshot,
        endSnapshot,
        peakSnapshot,
    }

    options.onProfile?.(profile)
    if (shouldLog) logGLExecutionProfile(profile, logLevel)

    return result
}

export function logGLMemorySnapshot(label: string = 'glMemory', snapshot?: GLMemorySnapshot): GLMemorySnapshot
{
    // Log GL memory snapshot counters and return the snapshot used for logging.
    const data = snapshot ?? getGLMemorySnapshot()
    console.log(
        `[${label}] numTensors=${data.numTensors} numDataBuffers=${data.numDataBuffers} numBytes=${data.numBytes} numBytesInGPU=${data.numBytesInGPU} numUsedTextures=${data.numUsedTextures} numFreeTextures=${data.numFreeTextures} numBytesAllocated=${data.numBytesAllocated} numBytesFree=${data.numBytesFree} numBytesUsed=${data.numBytesUsed} unreliable=${data.unreliable}`
    )

    return data
}

export function createGLPeakTracker(label: string = 'glPeak'): GLPeakTracker
{
    // Create a peak tracker and seed it with one initial snapshot.
    const tracker: GLPeakTracker = {
        label,
        samples: 0,
        peakNumTensors: 0,
        peakNumDataBuffers: 0,
        peakNumBytes: 0,
        peakNumBytesInGPU: 0,
        peakNumBytesAllocated: 0,
        peakNumBytesFree: 0,
        peakNumBytesUsed: 0,
        lastSnapshot: null,
    }

    return updateGLPeakTracker(tracker)
}

export function updateGLPeakTracker(tracker: GLPeakTracker): GLPeakTracker
{
    // Update running peaks from the latest memory snapshot.
    const snapshot = getGLMemorySnapshot()

    tracker.samples += 1
    tracker.peakNumTensors = Math.max(tracker.peakNumTensors, snapshot.numTensors)
    tracker.peakNumDataBuffers = Math.max(tracker.peakNumDataBuffers, snapshot.numDataBuffers)
    tracker.peakNumBytes = Math.max(tracker.peakNumBytes, snapshot.numBytes)
    tracker.peakNumBytesInGPU = Math.max(tracker.peakNumBytesInGPU, snapshot.numBytesInGPU)
    tracker.peakNumBytesAllocated = Math.max(tracker.peakNumBytesAllocated, snapshot.numBytesAllocated)
    tracker.peakNumBytesFree = Math.max(tracker.peakNumBytesFree, snapshot.numBytesFree)
    tracker.peakNumBytesUsed = Math.max(tracker.peakNumBytesUsed, snapshot.numBytesUsed)
    tracker.lastSnapshot = snapshot

    return tracker
}

export function logGLPeakTracker(tracker: GLPeakTracker, label?: string): GLPeakTracker
{
    // Log tracked peak memory/tensor counters and return tracker unchanged.
    const name = label ?? tracker.label
    console.log(
        `[${name}] samples=${tracker.samples} peakNumTensors=${tracker.peakNumTensors} peakNumDataBuffers=${tracker.peakNumDataBuffers} peakNumBytes=${tracker.peakNumBytes} peakNumBytesInGPU=${tracker.peakNumBytesInGPU} peakNumBytesAllocated=${tracker.peakNumBytesAllocated} peakNumBytesUsed=${tracker.peakNumBytesUsed}`
    )

    return tracker
}

export function logGLPeakCounters(tracker: GLPeakTracker, counters: GLPeakCounter[], label?: string): GLPeakTracker
{
    // Log only selected peak counters and return tracker unchanged.
    const name = label ?? tracker.label
    if (counters.length === 0)
    {
        console.log(`[${name}]`)
        return tracker
    }

    const payload = counters.map((counter) => `${counter}=${tracker[counter]}`).join(' ')
    console.log(`[${name}] ${payload}`)

    return tracker
}

export function createGLPeakTrackerLogger(label: string): (tracker: GLPeakTracker) => GLPeakTracker
{
    // Create a logger with a fixed label for repeated peak logs.
    return (tracker: GLPeakTracker) => logGLPeakTracker(tracker, label)
}

export function getGLTexturePoolStats(): GLTexturePoolStats | null
{
    // Inspect backend texture-pool counters for memory diagnostics/telemetry.
    const backend = getWebGLBackend()
    if (!backend) return null

    const textureManager = backend?.getTextureManager?.() ?? backend?.textureManager
    if (!textureManager) return null

    const numUsedTextures = textureManager?.getNumUsedTextures?.() ?? textureManager?.numUsedTextures ?? 0
    const numFreeTextures = textureManager?.getNumFreeTextures?.() ?? textureManager?.numFreeTextures ?? 0
    const numBytesAllocated = textureManager?.numBytesAllocated ?? textureManager?._numBytesAllocated ?? 0
    const numBytesFree = textureManager?.numBytesFree ?? textureManager?._numBytesFree ?? 0

    return {
        numUsedTextures,
        numFreeTextures,
        numBytesAllocated,
        numBytesFree,
    }
}

export function purgeGLFreeTexturePool(): GLPurgeFreeTexturePoolResult | null
{
    // Force-delete free textures in TFJS WebGL reuse pool to release GPU memory.
    const backend = getWebGLBackend()
    if (!backend) return null

    const textureManager = backend?.getTextureManager?.() ?? backend?.textureManager
    const gpgpu = backend?.getGPGPUContext?.() ?? backend?.gpgpu
    if (!textureManager || !gpgpu) return null

    const before = getGLTexturePoolStats()
    if (!before) return null

    const freeTextures = textureManager?.freeTextures
    if (!freeTextures || typeof freeTextures !== 'object')
    {
        return {
            deletedTextures: 0,
            freedBytes: 0,
            before,
            after: before,
        }
    }

    let deletedTextures = 0
    for (const key of Object.keys(freeTextures))
    {
        const bucket = freeTextures[key]
        if (!Array.isArray(bucket)) continue

        for (const tex of bucket)
        {
            const glTexture = tex?.texture
            if (glTexture != null)
            {
                gpgpu.deleteMatrixTexture(glTexture)
            }
            deletedTextures += 1
        }
    }

    const freedBytes = typeof textureManager?._numBytesFree === 'number' ? textureManager._numBytesFree : before.numBytesFree
    const prevAllocated = typeof textureManager?._numBytesAllocated === 'number' ? textureManager._numBytesAllocated : before.numBytesAllocated

    textureManager.freeTextures = {}
    if (typeof textureManager?.numFreeTextures === 'number') textureManager.numFreeTextures = 0
    if (typeof textureManager?._numBytesFree === 'number') textureManager._numBytesFree = 0
    if (typeof textureManager?._numBytesAllocated === 'number')
    {
        textureManager._numBytesAllocated = Math.max(0, prevAllocated - freedBytes)
    }

    const after = getGLTexturePoolStats() ?? {
        numUsedTextures: before.numUsedTextures,
        numFreeTextures: 0,
        numBytesAllocated: Math.max(0, before.numBytesAllocated - freedBytes),
        numBytesFree: 0,
    }

    return {
        deletedTextures,
        freedBytes,
        before,
        after,
    }
}

export function logGLTexturePoolStats(label: string = 'glPool'): GLTexturePoolStats | null
{
    // Log current texture-pool stats with a label and also return them for programmatic use.
    const stats = getGLTexturePoolStats()
    if (!stats) return null

    console.log(
        `[${label}] used=${stats.numUsedTextures} free=${stats.numFreeTextures} allocated=${stats.numBytesAllocated} freeBytes=${stats.numBytesFree}`
    )

    return stats
}
