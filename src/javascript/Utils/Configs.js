import EventEmitter from './EventEmitter'

/**
 * Configs
 * Manages global app state (interpolation, gradients, marching, skipping, toggles).
 */
export default class Configs extends EventEmitter 
{
    static Colormaps = Object.freeze([ 'parula', 'turbo', 'hsv', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'gray', 'bone', 'copper', 'pink', 'jet', 'pasteljet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis' ])

    static GradientsMethods = Object.freeze([
        'analytic',
        'trilinearSobel',
        'triquadraticBspline',
        'tricubicBspline',
    ])
    static MarchingMethods = Object.freeze([
        'cells',
        'traces',
    ])
    static SkippingStrategies = Object.freeze([
        'blocks',
        'blockGroups',
    ])
    static SkippingMethods = Object.freeze([
        'occupancy',
        'distanceMap',
    ])

    constructor() 
    {
        super()

        this.blockSize = 1
        this.downscaleFactor = 0.6  
        
        this.gradientsMethod = 'triquadraticBspline'
        this.marchingMethod = 'traces'
        this.skippingStrategy = 'blockGroups'
        this.skippingMethod = 'occupancy'
        this.colormap = 'viridis'

        this.bernsteinEnabled = false
        this.skippingEnabled = false

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
        if (key === 'intersectionTest' && !Configs.IntersectionTests.includes(value)) 
        {
            console.warn(`Invalid IntersectionTest: "${value}"`)
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
        if (key === 'isosurfaceValue' && (typeof value !== 'number' || value < 0 || value > 1)) 
        {
            console.warn(`isosurfaceValue must be in [0,1] (got ${value})`)
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
