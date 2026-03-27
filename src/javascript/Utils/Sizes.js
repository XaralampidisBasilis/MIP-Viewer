import EventEmitter from './EventEmitter'

/**
 * Sizes
 *
 * Tracks the browser viewport size and exposes a stable renderer pixel ratio.
 */
export default class Sizes extends EventEmitter
{
    minPixelRatio = 1
    maxPixelRatio = 1.5

    constructor()
    {
        super()

        this.updateSize = this.updateSize.bind(this)

        this.updateSize()
        window.addEventListener('resize', this.updateSize)
    }

    computePixelRatio()
    {
        const devicePixelRatio = window.devicePixelRatio || 1
        return Math.min(this.maxPixelRatio, Math.max(this.minPixelRatio, devicePixelRatio))
    }

    updateSize()
    {
        this.width = window.innerWidth
        this.height = window.innerHeight
        this.pixelRatio = this.computePixelRatio()

        this.trigger('resize')
    }

    destroy()
    {
        window.removeEventListener('resize', this.updateSize)

        this.width = null
        this.height = null
        this.pixelRatio = null

        console.log('Sizes destroyed')
    }
}
