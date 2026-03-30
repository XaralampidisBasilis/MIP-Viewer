import EventEmitter from './EventEmitter'

export default class PixelRatio extends EventEmitter
{
    constructor(experience)
    {
        super()

        this.experience = experience
        this.sizes = this.experience.sizes
        this.time = this.experience.time

        this.enabled = false 
        this.minPixelRatio = 0.5
        this.maxPixelRatioCap = 1.5
        this.targetPixels = 1920 * 1080

        this.targetFPS = 60
        this.adjustIntervalMs = 500
        this.smoothing = 0.15
        this.lowFPSThreshold = 0.95
        this.highFPSThreshold = 0.99
        this.maxDecreasePerStep = 0.2
        this.maxIncreasePerStep = 0.02
        this.epsilon = 0.01
        this.debug = false

        this.maxPixelRatio = this.getMaxPixelRatio()
        this.value = this.maxPixelRatio

        this.elapsedSinceAdjust = 0
        this.smoothedDelta = 1000 / this.targetFPS
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

        return Math.max(this.minPixelRatio, Math.min(this.maxPixelRatioCap, devicePixelRatio, budgetPixelRatio))
    }

    resetMetrics()
    {
        this.elapsedSinceAdjust = 0
        this.smoothedDelta = 1000 / this.targetFPS
    }

    getDecreaseScale(fpsRatio)
    {
        const minScale = 1 - this.maxDecreasePerStep

        return Math.max(minScale, Math.min(1, Math.sqrt(fpsRatio)))
    }

    getIncreaseScale(fpsRatio)
    {
        if(this.highFPSThreshold >= 1)
        {
            return 1
        }

        const normalizedHeadroom = Math.max(
            0,
            Math.min(1, (fpsRatio - this.highFPSThreshold) / (1 - this.highFPSThreshold))
        )

        return 1 + normalizedHeadroom * this.maxIncreasePerStep
    }

    update()
    {
        if(!this.enabled)
        {
            return false
        }

        if(!Number.isFinite(this.time.delta) || this.time.delta <= 0)
        {
            return false
        }

        this.elapsedSinceAdjust += this.time.delta
        this.smoothedDelta += (this.time.delta - this.smoothedDelta) * this.smoothing

        if(this.elapsedSinceAdjust < this.adjustIntervalMs)
        {
            return false
        }

        this.elapsedSinceAdjust = 0

        const fps = 1000 / this.smoothedDelta
        const fpsRatio = fps / this.targetFPS

        if(fpsRatio < this.lowFPSThreshold)
        {
            return this.apply(this.value * this.getDecreaseScale(fpsRatio))
        }

        if(fpsRatio > this.highFPSThreshold)
        {
            return this.apply(this.value * this.getIncreaseScale(fpsRatio))
        }

        return false
    }

    resize()
    {
        this.maxPixelRatio = this.getMaxPixelRatio()
        this.resetMetrics()

        return this.apply(this.enabled ? this.value : this.maxPixelRatio)
    }

    apply(pixelRatio)
    {
        if(!Number.isFinite(pixelRatio))
        {
            return false
        }

        const nextPixelRatio = this.clampValue(pixelRatio)

        if(Math.abs(nextPixelRatio - this.value) < this.epsilon)
        {
            return false
        }

        this.value = nextPixelRatio

        if(this.debug)
        {
            console.log(`pixelRatio: ${this.value}`)
        }

        this.trigger('rescale')

        return true
    }

    destroy()
    {
        this.experience = null
        this.sizes = null
        this.time = null
        this.value = null
        this.maxPixelRatio = null
    }
}