import EventEmitter from './EventEmitter'

export default class PixelRatio extends EventEmitter
{
    constructor(experience)
    {
        super()

        this.experience = experience
        this.sizes = this.experience.sizes
        this.time = this.experience.time

        this.minPixelRatio = 0.5
        this.maxPixelRatio = 1.5
        this.value = this.clampValue(window.devicePixelRatio)
        this.enabled = true

        this.targetFPS = 60
        this.adjustIntervalMs = 250
        this.decreaseFactor = 0.9
        this.increaseFactor = 1.05
        this.lowFPSThreshold = 0.95
        this.stableFPSThreshold = 0.985
        this.stableDurationMs = 1500
        this.elapsedSinceAdjust = 0
        this.framesSinceAdjust = 0
        this.stableElapsedMs = 0
    }

    clampValue(pixelRatio)
    {
        return Math.min(this.maxPixelRatio, Math.max(this.minPixelRatio, pixelRatio))
    }

    update()
    {
        if (!this.enabled)
        {
            return false
        }

        if (!Number.isFinite(this.time.delta) || this.time.delta <= 0)
        {
            return false
        }

        this.elapsedSinceAdjust += this.time.delta
        this.framesSinceAdjust += 1

        if (this.elapsedSinceAdjust < this.adjustIntervalMs)
        {
            return false
        }

        const elapsedSinceAdjust = this.elapsedSinceAdjust
        const fps = this.framesSinceAdjust * 1000 / elapsedSinceAdjust

        this.elapsedSinceAdjust = 0
        this.framesSinceAdjust = 0

        if (fps < this.targetFPS * this.lowFPSThreshold)
        {
            this.stableElapsedMs = 0
            return this.apply(this.value * this.decreaseFactor)
        }

        if (fps >= this.targetFPS * this.stableFPSThreshold)
        {
            this.stableElapsedMs += elapsedSinceAdjust

            if (this.stableElapsedMs >= this.stableDurationMs)
            {
                this.stableElapsedMs = 0
                return this.apply(this.value * this.increaseFactor)
            }

            return false
        }

        this.stableElapsedMs = 0
        return false
    }

    apply(pixelRatio)
    {
        const nextPixelRatio = this.clampValue(pixelRatio)

        if (Math.abs(nextPixelRatio - this.value) < 0.01)
        {
            return false
        }

        this.value = nextPixelRatio
        console.log(`pixelRatio: ${this.value}`)

        this.trigger('rescale')

        return true
    }

    resize()
    {

    }

    destroy()
    {
        this.experience = null
        this.sizes = null
        this.value = null
        this.minPixelRatio = null
        this.maxPixelRatio = null
        this.enabled = null
        this.elapsedSinceAdjust = null
        this.framesSinceAdjust = null
        this.stableElapsedMs = null
    }
}
