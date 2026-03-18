import EventEmitter from './Utils/EventEmitter'

/**
 * Configs
 * Manages global app state (interpolation, gradients, marching, skipping, toggles).
 */
export default class Configs extends EventEmitter 
{
    static Colormaps = Object.freeze([ 'parula', 'turbo', 'hsv', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'gray', 'bone', 'copper', 'pink', 'jet', 'pasteljet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis' ])

    static GradientsMethods = Object.freeze([
        'analytic',
        'triquadraticBspline',
    ])
    static MarchingMethods = Object.freeze([
        'cells',
        'traces',
    ])
    static SkippingStrategies = Object.freeze([
        'blocks',
        'groups',
    ])
    static SkippingMethods = Object.freeze([
        'binary',
        'distance',
    ])

    constructor() 
    {
        super()

        this.blockSize = 1

        this.downscaleEnabled = true
        this.downscaleFactor = 0.9
        
        this.gradientsMethod = 'triquadraticBspline'
        this.marchingMethod = 'cells'
        this.skippingStrategy = 'blocks'
        this.skippingMethod = 'distance'
        this.colormap = 'viridis'

        this.skippingEnabled = true
        this.debugEnabled = true
        this.statsEnabled = true
        this.discardingEnabled = true
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
        if (key === 'gradientsMethod' && !Configs.GradientsMethods.includes(value)) 
        {
            console.warn(`Invalid GradientsMethod: "${value}"`)
            return
        }
        if (key === 'marchingMethod' && !Configs.MarchingMethods.includes(value)) 
        {
            console.warn(`Invalid MarchingMethod: "${value}"`)
            return
        }
        if (key === 'skippingStrategy' && !Configs.SkippingStrategies.includes(value)) 
        {
            console.warn(`Invalid SkippingStrategy: "${value}"`)
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
