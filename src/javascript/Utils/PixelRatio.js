import EventEmitter from './EventEmitter'

export default class PixelRatio extends EventEmitter
{
    constructor(experience)
    {
        super()

        this.experience = experience
        this.configs = experience.configs
        this.time = experience.time

        this.enabled = this.configs.adaptivePixelRatioEnabled

        this.minValue = 0.5
        this.updateMaxValue()
        this.targetFps = 60
        this.adjustEveryMs = 300
        this.smoothing = 0.1

        this.lowThreshold = 0.9
        this.highThreshold = 0.99

        this.maxStepDown = 0.15
        this.maxStepUp = 0.03
        this.epsilon = 0.005

        this.value = this.clampValue(this.devicePixelRatio)

        this.elapsed = 0
        this.smoothedDelta = 1000 / this.targetFps
    }

    get devicePixelRatio()
    {
        return window.devicePixelRatio || 1
    }

    updateMaxValue()
    {
        this.maxValue = Math.min(this.devicePixelRatio, 1.25)
    }

    clampValue(value)
    {
        return Math.max(this.minValue, Math.min(this.maxValue, value))
    }

    reset()
    {
        this.elapsed = 0
        this.smoothedDelta = 1000 / this.targetFps
    }

    apply(next)
    {
        if (!Number.isFinite(next)) return false

        const clamped = this.clampValue(next)

        if (Math.abs(clamped - this.value) < this.epsilon)
        {
            return false
        }

        this.value = clamped
        console.log('Rescale:', this.value.toFixed(4))

        this.trigger('rescale')
        return true
    }

    update()
    {
        if (!this.enabled) return false
        if (!Number.isFinite(this.time.delta) || this.time.delta <= 0) return false

        this.updateMaxValue()
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
            const headroom = Math.min(1, Math.max(0, (ratio - this.highThreshold) / (1 - this.highThreshold)))

            const step = headroom * this.maxStepUp
            return this.apply(this.value + step)
        }

        return false
    }

    destroy()
    {
        this.experience = null
        this.time = null
        this.value = null
        this.maxValue = null
    }
}
