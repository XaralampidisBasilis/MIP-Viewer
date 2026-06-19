import * as THREE from 'three'
import { WebGPURenderer } from 'three/webgpu'
import Experience from './Experience'

/**
 * Renderer
 * 
 * Handles all WebGL rendering using Three.js.
 * Initializes the renderer instance, manages resizing, rendering, and cleanup.
 */
export default class Renderer
{
    constructor()
    {
        this.experience = new Experience()
        this.canvas = this.experience.canvas
        this.context = this.experience.context
        this.configs = this.experience.configs
        this.sizes = this.experience.sizes
        this.pixelRatio = this.experience.pixelRatio
        this.scene = this.experience.scene
        this.camera = this.experience.camera
        this.ready = false

        this.setInstance()
    }

    setInstance()
    {
        if (this.configs.renderBackend === 'webgpu' && globalThis.navigator?.gpu)
        {
            this.instance = new WebGPURenderer({
                canvas: this.canvas,
                antialias: false,
                depth: false,
                alpha: false,
            })

            this.instance.setClearColor('#211d20', 1)
            this.instance.shadowMap.enabled = false
            this.instance.setPixelRatio(this.pixelRatio.value)
            this.instance.setSize(this.sizes.width, this.sizes.height, false)
            return
        }

        const context = this.context ?? this.canvas.getContext('webgl2')
        if (!context) throw new Error('WebGL2 not supported by your browser or device.')
        this.context = context

        this.instance = new THREE.WebGLRenderer({
            canvas: this.canvas,
            context,
            antialias: false,
            depth: false,
        })       

        // Set clear color for the background
        this.instance.setClearColor('#211d20', 1)
        this.instance.shadowMap.enabled = false

        this.instance.setPixelRatio(this.pixelRatio.value)
        this.instance.setSize(this.sizes.width, this.sizes.height, false)
    }

    async start()
    {
        await this.instance?.init?.()
        this.ready = true
    }

    resize()
    {
        this.instance.setSize(this.sizes.width, this.sizes.height, false)    
    }

    rescale()
    {
        this.instance.setPixelRatio(this.pixelRatio.value)
        this.instance.setSize(this.sizes.width, this.sizes.height, false)
    }

    update()
    {
        if (!this.ready) return

        // Render the scene from the camera's perspective
        this.instance.render(this.scene, this.camera.instance)
    }

    destroy() 
    {
        // Dispose of the renderer
        if (this.instance) 
        {
            this.instance.dispose()
            this.instance = null
        }

        // Nullify references for cleanup
        this.experience = null
        this.canvas = null
        this.sizes = null
        this.scene = null
        this.camera = null
        this.ready = false

        console.log('Renderer destroyed')
    }
}
