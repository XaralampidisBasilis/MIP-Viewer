import * as THREE from 'three'
import Configs from './Configs'
import Sizes from './Utils/Sizes'
import Time from './Utils/Time'
import Mouse from './Utils/Mouse'
import Stats from './Utils/Stats'
import PixelRatio from './Utils/PixelRatio'
import Camera from './Camera'
import Renderer from './Renderer'
import World from './World/World'
import Resources from './Utils/Resources'
import Computes from './Computes/Computes'
import GUI from './GUI'
import sources from './sources'

export default class Experience
{
    static instance = null
    static started = false

    constructor(canvas, context)
    {
        // singleton
        if (Experience.instance) 
        {
            return Experience.instance
        }
        Experience.instance = this
        
        // Global access
        window.experience = this

        // Options
        this.canvas = canvas
        this.context = context

        // Setup
        this.configs = new Configs()
        this.sizes = new Sizes(canvas)
        this.time = new Time()
        this.pixelRatio = new PixelRatio(this)
        this.mouse = new Mouse()
        this.scene = new THREE.Scene()
        this.camera = new Camera()
        this.renderer = new Renderer()
        this.resources = new Resources(sources)
        this.computes = new Computes()
        this.world = new World()
        this.stats = new Stats(true)
        this.gui = new GUI()

        // Resources ready event
        this.resources.on('ready', () =>  this.start())

        // Time tick event
        this.time.on('tick', () => this.update())
        
        // Size resize event
        this.sizes.on('resize', () => this.resize())
    
        // Pixel ration adapt event
        this.pixelRatio.on('rescale', () => this.rescale())

        // Config change event
        this.configs.on('change', (event) => this.change(event))

        // Window refresh event
        window.addEventListener('beforeunload', () => this.destroy())
    }

    async start()
    {
        await this.computes.start()

        this.world.start()
        this.gui.start()

        Experience.started = true
    }

    
    updateWorld()
    {
        if (Experience.started) 
        {
            this.world.update()
        }
    }

    update()
    {
        this.pixelRatio.update()

        this.camera.update()
        this.updateWorld()
        this.renderer.update()
        this.stats.update()    
    }

    resize()
    {
        this.pixelRatio.resize()

        this.camera.resize()
        this.updateWorld()
        this.renderer.resize()
    }

    rescale()
    {
        this.renderer.rescale()
        this.camera.rescale()
        this.updateWorld()
    }

    async change(event)
    {
        // await this.computes.change(event)
        this.world.change(event)
    }

    destroy()
    {
        this.sizes.off('resize')
        this.time.off('tick')
        this.configs.off('change')

        // destroy components
        this.configs?.destroy()
        this.sizes?.destroy()
        this.time?.destroy()
        this.mouse?.destroy()
        this.world?.destroy()
        this.camera?.destroy()
        this.renderer?.destroy()
        this.pixelRatio?.destroy()
        this.computes?.destroy()
        this.gui?.destroy()

        // Nullify properties for cleanup
        this.configs = null
        this.sizes = null
        this.time = null
        this.mouse = null
        this.scene = null
        this.pixelRatio = null
        this.camera = null
        this.resources = null
        this.renderer = null
        this.world = null
        this.stats = null
        this.computes = null
        this.canvas = null

        // Clear the singleton instance
        Experience.instance = null

        console.log('Experience destroyed')
    }
}
