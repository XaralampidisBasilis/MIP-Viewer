import * as THREE from 'three'
import EventEmitter from './EventEmitter'

/**
 * Mouse
 * 
 * A utility class that tracks mouse movement and clicks.
 * Emits normalized device coordinates and screen position on events.
 */
export default class Mouse extends EventEmitter
{
    constructor()
    {
        super()

        this.screenPosition = new THREE.Vector2() // Screen mouse position
        this.ndcPosition = new THREE.Vector2() // Normalized mouse position (-1 to 1 range)

        // Bind event handlers to the instance
        this.onMouseMove = this.onMouseMove.bind(this)
        this.onMouseDown = this.onMouseDown.bind(this)
        this.onMouseUp = this.onMouseUp.bind(this)

        // Add event listeners
        window.addEventListener('mousemove', this.onMouseMove)
        window.addEventListener('mousedown', this.onMouseDown)
        window.addEventListener('mouseup', this.onMouseUp)
    }

    onMouseMove(event)
    {
        // Update mouse position
        this.screenPosition.x = event.clientX
        this.screenPosition.y = event.clientY

        // Update normalized mouse position
        this.ndcPosition.x = (this.screenPosition.x / window.innerWidth) * 2 - 1
        this.ndcPosition.y = (this.screenPosition.y / window.innerHeight) * 2 - 1
        this.ndcPosition.y *= - 1 // Flip Y-axis for NDC space

        // Emit the `move` event
        this.trigger('move', 
        {
            x: this.screenPosition.x,
            y: this.screenPosition.y,
            ndcX: this.ndcPosition.x,
            ndcY: this.ndcPosition.y,
        })
    }

    onMouseDown(event)
    {
        // Emit the `down` event with button information
        this.trigger('down', 
        {
            button: event.button,
            x: this.screenPosition.x,
            y: this.screenPosition.y,
            ndcX: this.ndcPosition.x,
            ndcY: this.ndcPosition.y,
        })
    }

    onMouseUp(event)
    {
        // Emit the `up` event with button information
        this.trigger('up', 
        {
            button: event.button,
            x: this.screenPosition.x,
            y: this.screenPosition.y,
            ndcX: this.ndcPosition.x,
            ndcY: this.ndcPosition.y,
        })
    }

    destroy() 
    {
        // Remove event listeners
        window.removeEventListener('mousemove', this.onMouseMove)
        window.removeEventListener('mousedown', this.onMouseDown)
        window.removeEventListener('mouseup', this.onMouseUp)

        // Clear properties for cleanup
        this.screenPosition = null
        this.ndcPosition = null

        console.log('Mouse destroyed')
    }
}
