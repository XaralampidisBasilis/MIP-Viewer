import mainWGSL from './main.wgsl?raw'
import structsWGSL from './structs/source'
import utilsWGSL from './utils/source'
import raycastWGSL from './raycast/source'
import samplingWGSL from './sampling/source'
import distanceWGSL from './distance/source'
import marchWGSL from './march/source'
import colormapWGSL from './colormap/source'
import debugWGSL from './debug.wgsl?raw'

export default [
    structsWGSL,
    utilsWGSL,
    raycastWGSL,
    samplingWGSL,
    distanceWGSL,
    marchWGSL,
    colormapWGSL,
    debugWGSL,
    mainWGSL,
].join('\n')
