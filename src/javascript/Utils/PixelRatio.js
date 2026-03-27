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
        this.targetPixels = 1920 * 1080
        this.maxPixelRatio = this.getMaxPixelRatio()
        this.value = this.maxPixelRatio
        this.enabled = false

        this.targetFPS = 60
        this.adjustIntervalMs = 500
        this.decreaseFactor = 0.9
        this.increaseFactor = 1.03
        this.lowFPSThreshold = 0.95
        this.highFPSThreshold = 0.99
        this.elapsedSinceAdjust = 0
        this.framesSinceAdjust = 0
    }

    clampValue(pixelRatio)
    {
        return Math.min(this.maxPixelRatio, Math.max(this.minPixelRatio, pixelRatio))
    }

    getMaxPixelRatio()
    {
        const devicePixelRatio = window.devicePixelRatio || 1
        const safeWidth = Math.max(1, this.sizes.width || window.innerWidth || 1)
        const safeHeight = Math.max(1, this.sizes.height || window.innerHeight || 1)
        const budgetPixelRatio = Math.sqrt(this.targetPixels / (safeWidth * safeHeight))

        return Math.max(this.minPixelRatio, Math.min(1.5, devicePixelRatio, budgetPixelRatio))
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
            return this.apply(this.value * this.decreaseFactor)
        }

        if (fps >= this.targetFPS * this.highFPSThreshold)
        {
            return this.apply(this.value * this.increaseFactor)
        }

        return false
    }

    resize()
    {
        this.maxPixelRatio = this.getMaxPixelRatio()
        this.elapsedSinceAdjust = 0
        this.framesSinceAdjust = 0
        const changed = this.apply(this.enabled ? this.value : this.maxPixelRatio, false)

        return changed
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

    destroy()
    {
        this.experience = null
        this.sizes = null
        this.value = null
        this.minPixelRatio = null
        this.maxPixelRatio = null
        this.targetPixels = null
        this.enabled = null
        this.elapsedSinceAdjust = null
        this.framesSinceAdjust = null
    }
}
