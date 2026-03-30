import EventEmitter from './EventEmitter'

export default class Sizes extends EventEmitter
{
    constructor(canvas)
    {
        super()

        this.canvas = canvas
        this.updateSize = this.updateSize.bind(this)

        this.updateSize()
        window.addEventListener('resize', this.updateSize)
    }

    updateSize()
    {
        const rectangle = this.canvas.getBoundingClientRect()

        this.width = Math.max(1, Math.round(rectangle.width))
        this.height = Math.max(1, Math.round(rectangle.height))

        this.trigger('resize')
    }

    destroy()
    {
        window.removeEventListener('resize', this.updateSize)

        this.canvas = null
        this.width = null
        this.height = null

        console.log('Sizes destroyed')
    }
}