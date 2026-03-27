import EventEmitter from './EventEmitter'

/**
 * Sizes
 *
 * Tracks the browser viewport size.
 */
export default class Sizes extends EventEmitter
{
    constructor()
    {
        super()

        this.updateSize = this.updateSize.bind(this)

        this.updateSize()
        window.addEventListener('resize', this.updateSize)
    }

    updateSize()
    {
        this.width = window.innerWidth
        this.height = window.innerHeight

        this.trigger('resize')
    }

    destroy()
    {
        window.removeEventListener('resize', this.updateSize)

        this.width = null
        this.height = null

        console.log('Sizes destroyed')
    }
}
