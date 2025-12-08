import EventEmitter from './EventEmitter'

/**
 * Time
 * 
 * A utility class that emits a `tick` event on each animation frame.
 * Tracks elapsed time and delta between frames. Useful for creating game loops or animation systems.
 */
export default class Time extends EventEmitter
{
    constructor()
    {
        super()

        // Setup
        this.start = Date.now()   // Timestamp when the instance was created
        this.current = this.start // Most recent frame timestamp
        this.elapsed = 0          // Total time since start in milliseconds
        this.delta = 16           // Time between frames (initialized to ~60fps)

        // Control for animation frame ID
        this.animationFrameId = null

        // Start the tick loop
        this.tick = this.tick.bind(this) // Bind the tick method
        this.animationFrameId = window.requestAnimationFrame(this.tick)
    }

    tick()
    {
        const currentTime = Date.now()
        this.delta = currentTime - this.current // Time since last frame
        this.current = currentTime
        this.elapsed = this.current - this.start

        // Emit the `tick` event
        this.trigger('tick', {
            elapsed: this.elapsed,
            delta: this.delta
        })

        // Continue the loop
        this.animationFrameId = window.requestAnimationFrame(this.tick)
    }

    destroy()
    {
        // Cancel the animation frame loop
        if (this.animationFrameId !== null)
        {
            window.cancelAnimationFrame(this.animationFrameId)
            this.animationFrameId = null
        }

        // Nullify properties for cleanup
        this.start = null
        this.current = null
        this.elapsed = null
        this.delta = null

        console.log('Time destroyed')
    }
}
