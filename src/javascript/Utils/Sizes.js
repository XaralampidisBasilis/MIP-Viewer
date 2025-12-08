import EventEmitter from './EventEmitter'

/**
 * Sizes
 * 
 * A utility class to track and respond to changes in the browser window size.
 * Extends EventEmitter to notify subscribers when the window is resized.
 */
export default class Sizes extends EventEmitter 
{
    targetPixels = 1500000
    minPixelRatio = 0.5
    maxPixelRatio = 1.0

    constructor() 
    {
        super()

        // Setup initial dimensions
        this.width = window.innerWidth
        this.height = window.innerHeight
        this.pixelRatio = this.computePixelRatio(this.width, this.height) 

        // Bind resize event
        this.onResize = this.onResize.bind(this)
        window.addEventListener('resize', this.onResize)
    }

    computePixelRatio(width, height)
    {
        const devicePR = window.devicePixelRatio || 1

        // Pixel-budget-based DPR so: width*height*(dpr^2) ~= targetPixels
        const budgetPR = Math.sqrt(this.targetPixels / Math.max(1, width * height))

        // Respect both device DPR and quality bounds
        const capped = Math.min(devicePR, this.maxPixelRatio, budgetPR)
        return Math.max(this.minPixelRatio, capped)
    }


    onResize() 
    {
        // Update dimensions
        this.width = window.innerWidth
        this.height = window.innerHeight
        this.pixelRatio = this.computePixelRatio(this.width, this.height)
        console.log(this.pixelRatio)

        // Emit the `resize` event with updated values
        this.trigger('resize', 
        {
            width: this.width,
            height: this.height,
            pixelRatio: this.pixelRatio
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

        console.log('Sizes destroyed')
    }
}
