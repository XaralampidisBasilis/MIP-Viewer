import EventEmitter from './EventEmitter'

export default class PixelRatio extends EventEmitter
{
    constructor(experience)
    {
        super()

        this.experience = experience
        this.time = experience.time

        this.enabled = true

        this.minValue = 0.5
        this.maxValue = 1.5
        this.targetFps = 60
        this.adjustEveryMs = 500
        this.smoothing = 0.15

        this.lowThreshold = 0.95
        this.highThreshold = 0.99

        this.maxStepDown = 0.2
        this.maxStepUp = 0.02
        this.epsilon = 0.01

        this.maxAllowed = this.computeMaxAllowed()
        this.value = this.maxAllowed

        this.elapsed = 0
        this.smoothedDelta = 1000 / this.targetFps
    }

    get devicePixelRatio()
    {
        return window.devicePixelRatio || 1
    }

    computeMaxAllowed()
    {
        return Math.max(
            this.minValue,
            Math.min(this.maxValue, this.devicePixelRatio)
        )
    }

    clamp(value)
    {
        return Math.max(this.minValue, Math.min(this.maxAllowed, value))
    }

    reset()
    {
        this.elapsed = 0
        this.smoothedDelta = 1000 / this.targetFps
    }

    apply(next)
    {
        if (!Number.isFinite(next)) return false

        const clamped = this.clamp(next)

        if (Math.abs(clamped - this.value) < this.epsilon)
        {
            return false
        }

        this.value = clamped
        console.log('pixelRatio:', this.value)

        this.trigger('rescale')
        return true
    }

    update()
    {
        if (!this.enabled) return false
        if (!Number.isFinite(this.time.delta) || this.time.delta <= 0) return false

        this.elapsed += this.time.delta
        this.smoothedDelta += (this.time.delta - this.smoothedDelta) * this.smoothing

        if (this.elapsed < this.adjustEveryMs) return false
        this.elapsed = 0

        const fps = 1000 / this.smoothedDelta
        const ratio = fps / this.targetFps

        if (ratio < this.lowThreshold)
        {
            const scale = Math.max(1 - this.maxStepDown, Math.sqrt(ratio))
            return this.apply(this.value * scale)
        }

        if (ratio > this.highThreshold)
        {
            const headroom = Math.min(
                1,
                Math.max(0, (ratio - this.highThreshold) / (1 - this.highThreshold))
            )

            const scale = 1 + headroom * this.maxStepUp
            return this.apply(this.value * scale)
        }

        return false
    }

    resize()
    {
        this.maxAllowed = this.computeMaxAllowed()
        this.reset()

        if (this.value > this.maxAllowed)
        {
            return this.apply(this.maxAllowed)
        }

        return false
    }

    destroy()
    {
        this.experience = null
        this.time = null
        this.value = null
        this.maxAllowed = null
    }
}