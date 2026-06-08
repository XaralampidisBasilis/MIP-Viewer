import EventEmitter from './Utils/EventEmitter'

/**
 * Configs
 * Manages global app state (interpolation, gradients, marching, skipping, toggles).
 */
export default class Configs extends EventEmitter 
{
    static Colormaps = Object.freeze([ 'parula', 'turbo', 'hsv', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'gray', 'bone', 'copper', 'pink', 'jet', 'pasteljet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis' ])

    static MarchingMethods = Object.freeze([
        'cells',
        'traces',
    ])
    static SkippingMethods = Object.freeze([
        'shadow',
        'distance',
    ])
    static DistanceVariations = Object.freeze([
        '1bit',
        '5bit',
        '8bit',
        '10bit',
    ])

    constructor() 
    {
        super()

        this.adaptivePixelRatioEnabled = false
        
        this.downscaleEnabled = false
        this.downscaleFactor = 0.8

        this.errorTolerance = 0.01
        this.blockSize = 1
        
        this.distanceVariation = '1bit'
        this.marchingMethod = 'cells'
        this.skippingMethod = 'distance'
        this.colormap = 'viridis'

        this.skippingEnabled = true
        this.debugEnabled = true
    }

    set(key, value) 
    {
        this.check(key, value)

        if (key in this) 
        { 
            const newValue = value
            const oldValue = this[key] 
            this[key] = newValue 

            this.trigger('change', [{ key, oldValue, newValue }]) 
        } 
        else 
        { 
            console.warn(`Unknown config key: ${key}`)
        }
    }

    check(key, value)
    {
        if (key === 'colormap' && !Configs.Colormaps.includes(value)) 
        {
            console.warn(`Invalid Colormap: "${value}"`)
            return
        }
        if (key === 'marchingMethod' && !Configs.MarchingMethods.includes(value)) 
        {
            console.warn(`Invalid MarchingMethod: "${value}"`)
            return
        }
        if (key === 'skippingMethod' && !Configs.SkippingMethods.includes(value)) 
        {
            console.warn(`Invalid SkippingMethod: "${value}"`)
            return
        }
        if (key === 'blockSize' && (!Number.isInteger(value) || value <= 0)) 
        {
            console.warn(`blockSize must be a positive integer (got ${value})`)
            return
        }
        if (key === 'downscaleFactor' && (typeof value !== 'number' || value <= 0 || value > 1)) 
        {
            console.warn(`downscaleFactor must be in (0,1] (got ${value})`)
            return
        }
        if (key.endsWith('Enabled') && typeof value !== 'boolean') 
        {
            console.warn(`${key} must be boolean (got ${typeof value})`)
            return
        }
    }

    get(key) 
    {
        return (key in this) ? this[key] : null
    }

    destroy() 
    {
        console.log('Configs destroyed')
    }
}
