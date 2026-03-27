import EventEmitter from './EventEmitter'

export default class PixelRatio extends EventEmitter
{
    constructor(experience)
    {
        this.experience = experience
        this.sizes = this.experience.sizes
        this.configs = this.experience.configs

        this.value = 1
        this.minPixelRatio = 0.5
        this.maxPixelRatioCap = 1.5
        this.maxPixelRatio = this.computeMaxPixelRatio()
        this.enabled = this.configs.adaptivePixelRatioEnabled

        this.targetFPS = 60
        this.adjustIntervalMs = 250
        this.decreaseFactor = 0.9
        this.increaseFactor = 1.05
        this.lowFPSThreshold = 0.95
        this.elapsedSinceAdjust = 0
        this.framesSinceAdjust = 0
    }

    computeMaxPixelRatio()
    {
        const devicePixelRatio = window.devicePixelRatio || 1
        return Math.min(this.maxPixelRatioCap, Math.max(this.minPixelRatio, devicePixelRatio))
    }

    clamp(pixelRatio)
    {
        return Math.min(this.maxPixelRatio, Math.max(this.minPixelRatio, pixelRatio))
    }

    apply(pixelRatio)
    {
        const nextPixelRatio = this.clamp(pixelRatio)

        if (Math.abs(nextPixelRatio - this.experience.pixelRatio) < 0.01)
        {
            return false
        }

        this.experience.pixelRatio = nextPixelRatio

        console.log(`pixelRatio: ${this.experience.pixelRatio}`)

        return true
    }

    initialize()
    {
        this.experience.pixelRatio = this.maxPixelRatio
    }

    setEnabled(enabled)
    {
        this.enabled = enabled
        this.elapsedSinceAdjust = 0
        this.framesSinceAdjust = 0

        if (!enabled)
        {
            this.maxPixelRatio = this.computeMaxPixelRatio()
            return this.apply(this.maxPixelRatio)
        }

        return false
    }

    resize()
    {
        this.maxPixelRatio = this.computeMaxPixelRatio()
        const nextPixelRatio = this.enabled
            ? this.clamp(this.experience.pixelRatio)
            : this.maxPixelRatio

        return this.apply(nextPixelRatio)
    }

    update(delta)
    {
        if (!this.enabled)
        {
            return false
        }

        if (!Number.isFinite(delta) || delta <= 0)
        {
            return false
        }

        this.elapsedSinceAdjust += delta
        this.framesSinceAdjust += 1

        if (this.elapsedSinceAdjust < this.adjustIntervalMs)
        {
            return false
        }

        const fps = this.framesSinceAdjust * 1000 / this.elapsedSinceAdjust

        this.elapsedSinceAdjust = 0
        this.framesSinceAdjust = 0

        if (fps < this.targetFPS * this.lowFPSThreshold)
        {
            return this.apply(this.experience.pixelRatio * this.decreaseFactor)
        }

        if (fps > this.targetFPS)
        {
            return this.apply(this.experience.pixelRatio * this.increaseFactor)
        }

        return false
    }

    change(event)
    {
        if (event.key === 'adaptivePixelRatioEnabled')
        {
            return this.setEnabled(this.configs.adaptivePixelRatioEnabled)
        }

        return false
    }

    destroy()
    {
        this.experience = null
        this.sizes = null
        this.configs = null
        this.maxPixelRatio = null
        this.enabled = null
        this.elapsedSinceAdjust = null
        this.framesSinceAdjust = null
    }
}
