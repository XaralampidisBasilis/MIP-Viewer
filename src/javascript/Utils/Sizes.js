import EventEmitter from './EventEmitter'

/**
 * Sizes
 * 
 * A utility class to track and respond to changes in the browser window size.
 * Extends EventEmitter to notify subscribers when the window is resized.
 */
export default class Sizes extends EventEmitter 
{
    // Static pixel budget used to estimate a safe upper DPR from the viewport size.
    targetPixels = 1920 * 1080 // 1920*1080, 1280*800, 800*600, 640*480
    minPixelRatio = 0.5
    maxPixelRatio = 1.5
    pixelRatioStep = 0.05

    // Runtime adaptation parameters. We drop quality faster than we recover it.
    adaptiveEnabled = true
    targetFrameTimeMs = 1000 / 60
    adaptIntervalMs = 250
    adaptStableThresholdMs = 17.0
    adaptDownThresholdMs = 18.5
    adaptUpAfterStableMs = 1500
    adaptDownStep = 0.1
    adaptUpStep = 0.05
    adaptEpsilon = 0.05
    smoothingFactor = 0.1

    constructor() 
    {
        super()

        // Setup initial dimensions
        this.width = window.innerWidth
        this.height = window.innerHeight

        // `targetPixelRatio` is the budget-based ceiling for the current window size.
        this.targetPixelRatio = this.computePixelRatio(this.width, this.height)

        // Start from the budget-based target, then let runtime adaptation move below it if needed.
        this.pixelRatio = this.targetPixelRatio
        console.log(`pixelRatio: ${this.pixelRatio}`)

        // Exponential smoothing keeps one slow frame from immediately changing resolution.
        this.smoothedFrameTime = 16
        this.lastAdaptTime = 0
        this.stableSinceElapsed = 0

        // Bind resize event
        this.onResize = this.onResize.bind(this)
        window.addEventListener('resize', this.onResize)
    }

    computePixelRatio(width, height)
    {        
        const safeWidth = Math.max(1, width)
        const safeHeight = Math.max(1, height)
        const devicePR = this.clampPixelRatio(window.devicePixelRatio || 1)

        // Estimate the DPR that keeps the total rendered pixels near our target budget:
        // width * height * dpr^2 ~= targetPixels
        const budgetPR = Math.sqrt(this.targetPixels / (safeWidth * safeHeight))

        // Never exceed the device DPR or our app-specific cap.
        const capped = Math.min(devicePR, this.maxPixelRatio, budgetPR)

        // Snap to stable steps so small resizes do not constantly churn the renderer DPR.
        return this.quantizePixelRatio(this.clampPixelRatio(capped))
    }

    clampPixelRatio(value, max = this.maxPixelRatio)
    {
        return Math.max(this.minPixelRatio, Math.min(max, value))
    }

    quantizePixelRatio(value)
    {
        const snapped = Math.floor((value + 1e-6) / this.pixelRatioStep) * this.pixelRatioStep
        return Number(this.clampPixelRatio(snapped).toFixed(2))
    }

    emitResize()
    {
        this.trigger('resize', 
        {
            width: this.width,
            height: this.height,
            pixelRatio: this.pixelRatio
        })
    }

    /**
     * Adjusts the live render pixel ratio from recent frame timing.
     * Lowers DPR quickly when frame time drifts away from the 60 FPS target,
     * then cautiously probes upward after the renderer has stayed stable near
     * that target for a while. The result always stays within the size-based
     * budget computed for the current viewport.
     */
    updateAdaptivePixelRatio(delta, elapsed)
    {
        if (!this.adaptiveEnabled || !Number.isFinite(delta) || delta <= 0)
        {
            return
        }

        // Smooth the incoming frame time so quality changes respond to trends, not spikes.
        this.smoothedFrameTime += (delta - this.smoothedFrameTime) * this.smoothingFactor

        const isStableAtTarget = this.smoothedFrameTime <= this.adaptStableThresholdMs

        if (isStableAtTarget)
        {
            if (this.stableSinceElapsed === 0)
            {
                this.stableSinceElapsed = elapsed
            }
        }
        else
        {
            this.stableSinceElapsed = 0
        }

        // Only reconsider DPR every few hundred milliseconds to avoid resize thrashing.
        if (elapsed - this.lastAdaptTime < this.adaptIntervalMs)
        {
            return
        }

        this.lastAdaptTime = elapsed

        let nextPixelRatio = this.pixelRatio

        if (this.smoothedFrameTime > this.adaptDownThresholdMs)
        {
            nextPixelRatio -= this.adaptDownStep
            this.stableSinceElapsed = 0
        }
        else if (
            this.pixelRatio < this.targetPixelRatio &&
            this.stableSinceElapsed > 0 &&
            elapsed - this.stableSinceElapsed >= this.adaptUpAfterStableMs
        )
        {
            nextPixelRatio += this.adaptUpStep
            // Require another full stable window before increasing DPR again.
            this.stableSinceElapsed = elapsed
        }

        // Runtime adaptation may lower DPR, but it should never rise above the size-based budget.
        nextPixelRatio = this.quantizePixelRatio(this.clampPixelRatio(nextPixelRatio, this.targetPixelRatio))

        // Ignore tiny changes that would cause extra resizes without a visible benefit.
        if (Math.abs(nextPixelRatio - this.pixelRatio) < this.adaptEpsilon)
        {
            return
        }

        this.pixelRatio = nextPixelRatio
        console.log(`adaptive pixelRatio: ${this.pixelRatio.toFixed(2)} (${this.smoothedFrameTime.toFixed(2)} ms)`)
        this.emitResize()
    }

    onResize() 
    {
        // Update dimensions
        this.width = window.innerWidth
        this.height = window.innerHeight
        this.targetPixelRatio = this.computePixelRatio(this.width, this.height)
        
        // Keep the current adaptive DPR if it is already lower; otherwise clamp to the new ceiling.
        this.pixelRatio = Math.min(this.pixelRatio, this.targetPixelRatio)
        this.stableSinceElapsed = 0
        console.log(`pixelRatio: ${this.pixelRatio}`)

        // Emit the `resize` event with updated values
        this.emitResize()
    }

    destroy() 
    {
        // Remove the resize event listener
        window.removeEventListener('resize', this.onResize)

        // Nullify properties for cleanup
        this.width = null
        this.height = null
        this.pixelRatio = null
        this.targetPixelRatio = null
        this.smoothedFrameTime = null
        this.lastAdaptTime = null
        this.stableSinceElapsed = null

        console.log('Sizes destroyed')
    }
}
