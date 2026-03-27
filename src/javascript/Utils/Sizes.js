import EventEmitter from './EventEmitter'

/**
 * Sizes
 * 
 * A utility class to track and respond to changes in the browser window size.
 * Extends EventEmitter to notify subscribers when the window is resized.
 */
export default class Sizes extends EventEmitter 
{
    // Raymarching is typically fill-rate bound, so the viewport DPR ceiling should be conservative.
    referencePixels = 1600 * 900
    pixelRatioSafety = 0.9
    minPixelRatio = 0.5
    maxPixelRatio = 1.5
    pixelRatioStep = 0.05

    // Runtime adaptation aims to stay near 60 FPS and uses slower quality recovery than quality drop.
    adaptiveEnabled = true
    targetFrameTimeMs = 1000 / 60
    stableFrameSlackMs = 0.35
    slowFrameSlackMs = 1.5
    adaptIntervalMs = 250
    adaptUpAfterStableMs = 1500
    adaptDownStep = 0.1
    adaptUpStep = 0.05
    adaptEpsilon = 0.001
    smoothingFactor = 0.1

    constructor() 
    {
        super()

        // Setup initial dimensions
        this.width = window.innerWidth
        this.height = window.innerHeight

        // `targetPixelRatio` is the viewport-dependent quality ceiling for the current window size.
        this.targetPixelRatio = this.computePixelRatio(this.width, this.height)

        // Start from the safe ceiling, then let runtime adaptation lower or probe around it.
        this.pixelRatio = this.targetPixelRatio
        this.updateRenderSize()
        console.log(`pixelRatio: ${this.pixelRatio}`)

        // Exponential smoothing keeps one slow frame from immediately changing resolution.
        this.smoothedFrameTime = this.targetFrameTimeMs
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
        const devicePR = Math.max(1, window.devicePixelRatio || 1)

        // Estimate a safe DPR ceiling for fragment-heavy raymarching work:
        // width * height * dpr^2 ~= referencePixels, then keep a bit of headroom.
        const viewportBudgetPR = Math.sqrt(this.referencePixels / (safeWidth * safeHeight))
        const budgetPR = viewportBudgetPR * this.pixelRatioSafety

        // Never exceed the device DPR or our app-specific cap.
        const capped = Math.min(devicePR, this.maxPixelRatio, budgetPR)

        // Snap downward so the computed ceiling stays conservative.
        return this.quantizePixelRatio(this.clampPixelRatio(capped), 'down')
    }

    clampPixelRatio(value, max = this.maxPixelRatio)
    {
        return Math.max(this.minPixelRatio, Math.min(max, value))
    }

    quantizePixelRatio(value, mode = 'nearest')
    {
        return value
        
        const scaled = value / this.pixelRatioStep
        const snapped =
            mode === 'down'
                ? Math.floor(scaled + 1e-6) * this.pixelRatioStep
                : Math.round(scaled) * this.pixelRatioStep

        return Number(this.clampPixelRatio(snapped).toFixed(2))
    }

    getStableFrameThreshold()
    {
        return this.targetFrameTimeMs + this.stableFrameSlackMs
    }

    getSlowFrameThreshold()
    {
        return this.targetFrameTimeMs + this.slowFrameSlackMs
    }

    updateRenderSize()
    {
        // Match Three.js viewport rounding so the camera aspect can follow the real render target.
        this.renderWidth = Math.max(1, Math.round(this.width * this.pixelRatio))
        this.renderHeight = Math.max(1, Math.round(this.height * this.pixelRatio))
    }

    applyPixelRatio(nextPixelRatio, reason)
    {
        if (Math.abs(nextPixelRatio - this.pixelRatio) < this.adaptEpsilon)
        {
            return
        }

        this.pixelRatio = nextPixelRatio
        this.updateRenderSize()
        console.log(`${reason} pixelRatio: ${this.pixelRatio.toFixed(2)} (${this.smoothedFrameTime.toFixed(2)} ms)`)
        this.emitResize({ viewportChanged: false, pixelRatioChanged: true, reason })
    }

    emitResize({ viewportChanged = true, pixelRatioChanged = true, reason = 'resize' } = {})
    {
        this.trigger('resize', [{
            width: this.width,
            height: this.height,
            pixelRatio: this.pixelRatio,
            renderWidth: this.renderWidth,
            renderHeight: this.renderHeight,
            viewportChanged,
            pixelRatioChanged,
            reason,
        }])
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

        const isStableAtTarget = this.smoothedFrameTime <= this.getStableFrameThreshold()

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

        if (this.smoothedFrameTime > this.getSlowFrameThreshold())
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
        this.applyPixelRatio(nextPixelRatio, 'adaptive')
    }

    onResize() 
    {
        // Update dimensions
        const previousWidth = this.width
        const previousHeight = this.height
        const previousPixelRatio = this.pixelRatio

        this.width = window.innerWidth
        this.height = window.innerHeight
        
        const previousTargetPixelRatio = this.targetPixelRatio
        this.targetPixelRatio = this.computePixelRatio(this.width, this.height)

        // Clamp down immediately when the viewport grows, but let runtime adaptation probe up again when it shrinks.
        if (this.targetPixelRatio < this.pixelRatio)
        {
            this.pixelRatio = this.targetPixelRatio
        }
        else if (this.targetPixelRatio > previousTargetPixelRatio && this.pixelRatio < this.targetPixelRatio)
        {
            this.pixelRatio = this.quantizePixelRatio(
                this.clampPixelRatio(this.pixelRatio + this.adaptUpStep, this.targetPixelRatio)
            )
        }

        this.updateRenderSize()
        this.stableSinceElapsed = 0
        console.log(`pixelRatio: ${this.pixelRatio}`)

        // Emit the `resize` event with updated values
        this.emitResize({
            viewportChanged: this.width !== previousWidth || this.height !== previousHeight,
            pixelRatioChanged: this.pixelRatio !== previousPixelRatio,
            reason: 'viewport',
        })
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
        this.renderWidth = null
        this.renderHeight = null
        this.smoothedFrameTime = null
        this.lastAdaptTime = null
        this.stableSinceElapsed = null

        console.log('Sizes destroyed')
    }
}
