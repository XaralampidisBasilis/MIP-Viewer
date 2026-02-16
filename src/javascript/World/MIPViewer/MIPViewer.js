import * as THREE from 'three'
import Experience from '../../Experience'
import EventEmitter from '../../Utils/EventEmitter'
import Configs from '../../Utils/Configs'
import ISOMaterial from './MIPMaterial'

export default class MIPViewer extends EventEmitter
{
    static instance = null

    constructor()
    {
        super()
   
        if (MIPViewer.instance) 
        {
            return MIPViewer.instance
        }
        MIPViewer.instance = this

        this.experience = new Experience()
        this.scene = this.experience.scene
        this.resources = this.experience.resources
        this.renderer = this.experience.renderer
        this.computes = this.experience.computes
        this.debug = this.experience.debug
        this.configs = this.experience.configs

        this.setMesh()
    }

    start()
    {
        this.setMaterial()
        this.size = this.computes.volumeMap.size
        this.mesh.scale.copy(this.size)
        console.log(this)
    }

    setMesh()
    {   
        this.material = ISOMaterial()
        this.uniforms = this.material.uniforms
        this.defines = this.material.defines

        this.geometry = new THREE.BoxGeometry(1, 1, 1)
        this.mesh = new THREE.Mesh(this.geometry, this.material)
       
    }

    setMaterial()
    {
        const translation = new THREE.Matrix4().makeTranslation(0.5, 0.5, 0.5)
        const scale = new THREE.Matrix4().makeScale(...this.computes.volumeMap.dimensions)
        this.material.uniforms.uCustomModelMatrix.value.multiplyMatrices(scale, translation)

        this.setDefinesEnablers()
        this.setDefinesMethods()
        this.setDefinesIterators()
        this.setUniformsTextures()
        this.setUniformsVolume()
        this.setUniformsDebug()
        this.setUniformsShading()

    }

    setDefinesEnablers()
    {
        const configs = this.configs
        const defines = this.material.defines
        defines.BERNSTEIN_ENABLED = Number(configs.bernsteinEnabled)
        defines.SKIPPING_ENABLED = Number(configs.skippingEnabled)
        defines.DEBUG_ENABLED = Number(configs.debugEnabled)
        defines.DISCARDING_ENABLED = Number(configs.discardingEnabled)
        this.material.needsUpdate = true
    }

    setDefinesMethods()
    {
        const configs = this.configs
        const defines = this.material.defines
        defines.MARCHING_METHOD = Configs.MarchingMethods.findIndex((x) => x === configs.marchingMethod)
        defines.SKIPPING_STRATEGY = Configs.SkippingStrategies.findIndex((x) => x === configs.skippingStrategy)
        defines.SKIPPING_METHOD = Configs.SkippingMethods.findIndex((x) => x === configs.skippingMethod)
        defines.GRADIENTS_METHOD = Configs.GradientsMethods.findIndex((x) => x === configs.gradientsMethod)  
        this.material.needsUpdate = true
    }

    setDefinesIterators()
    {        
        const defines = this.material.defines
        defines.MAX_CELLS = this.computes.volumeMap.dimensions.toArray().reduce((y, x) => y + x, 0)
        defines.MAX_BLOCKS = this.computes.shadowMap.dimensions.toArray().reduce((y, x) => y + x, 0)
        defines.MAX_TRACES = Math.ceil(this.computes.volumeMap.dimensions.length() * 2)
        defines.MAX_CELLS_IN_BLOCK = Math.ceil(this.configs.blockSize * 3)
        defines.MAX_TRACES_IN_BLOCK = Math.ceil(this.configs.blockSize * Math.sqrt(3) * 2)
        // defines.MAX_GROUPS = Math.ceil(defines.MAX_CELLS / defines.MAX_CELLS_IN_BLOCK)
        // defines.MAX_BLOCKS_IN_GROUP = Math.ceil(defines.MAX_BLOCKS / defines.MAX_GROUPS)
        defines.MAX_GROUPS = Math.ceil(defines.MAX_TRACES / defines.MAX_TRACES_IN_BLOCK)
        defines.MAX_BLOCKS_IN_GROUP = Math.ceil(defines.MAX_BLOCKS / defines.MAX_GROUPS)
        this.material.needsUpdate = true

        console.log(defines)
    }

    setUniformsTextures()
    {
        const uniforms = this.material.uniforms
        uniforms.u_textures.value.volume_map = this.computes.volumeMap.texture
        uniforms.u_textures.value.shadow_map = this.computes.shadowMap.texture
        // uniforms.u_textures.value.distance_map = this.computes.distanceMap.texture
    }

    setUniformsVolume()
    {
        const uniforms = this.material.uniforms
        uniforms.u_volume.value.dimensions.copy(this.computes.volumeMap.dimensions)
        uniforms.u_volume.value.spacing.copy(this.computes.volumeMap.spacing)
        uniforms.u_volume.value.spacing_normalized.copy(this.computes.volumeMap.spacing).normalize()
        uniforms.u_volume.value.block_size = this.configs.blockSize
        uniforms.u_volume.value.blocked_dimensions.copy(this.computes.shadowMap.dimensions)
        uniforms.u_volume.value.inv_dimensions.fromArray(uniforms.u_volume.value.dimensions.toArray().map(x => 1/x))
    }

    setUniformsShading()
    {
        const uniforms = this.material.uniforms
        uniforms.u_shading.value.colormap = Configs.Colormaps.findIndex((x) => x === this.configs.colormap)
    }

    setUniformsDebug()
    {
        const uniforms = this.material.uniforms
        uniforms.u_debug.value.max_groups = this.material.defines.MAX_GROUPS
        uniforms.u_debug.value.max_blocks = this.material.defines.MAX_BLOCKS_IN_GROUP 
        uniforms.u_debug.value.max_cells  = this.material.defines.MAX_CELLS_IN_BLOCK  
    }

    change(event)
    {
        if      (event.key === 'blockSize'          ) this.onChangeBlockSize(event)
        else if (event.key === 'downscaleFactor'    ) this.onChangeDownscaleFactor(event)
        else if (event.key === 'skippingStrategy'   ) this.onChangeSkippingStrategy(event)
        else if (event.key === 'skippingMethod'     ) this.onChangeSkippingMethod(event)
        else if (event.key === 'gradientsMethod'    ) this.onChangeGradientsMethod(event)
        else if (event.key === 'marchingMethod'     ) this.onChangeMarchingMethod(event)
        else if (event.key === 'skippingEnabled'    ) this.onChangeSkippingEnabled(event)
        else if (event.key === 'colormap'           ) this.onChangeColormap(event)
        
        console.log(this)
    }

    onChangeBlockSize(event)
    {
        const uniforms = this.material.uniforms
        uniforms.u_volume.value.block_size = this.configs.blockSize
        uniforms.u_volume.value.blocked_dimensions.copy(this.computes.shadowMap.dimensions)
        uniforms.u_textures.value.shadow_map.dispose()
        uniforms.u_textures.value.shadow_map = this.computes.shadowMap.texture    
        uniforms.u_textures.value.distance_map.dispose()
        uniforms.u_textures.value.distance_map = this.computes.distanceMap.texture
        this.setDefinesIterators()
    }

    onChangeDownscaleFactor(event)
    {
        const uniforms = this.material.uniforms
        uniforms.u_textures.value.volume_map.dispose()
        uniforms.u_textures.value.shadow_map.dispose()
        uniforms.u_textures.value.distance_map.dispose()

        this.material.dispose()
        this.setMaterial()
    }

    onChangeSkippingStrategy(event)
    {
        this.material.defines.SKIPPING_STRATEGY = Configs.SkippingStrategies.findIndex((x) => x === this.configs.skippingStrategy)
        this.material.needsUpdate = true
    }

    onChangeSkippingMethod(event)
    {
        this.material.uniforms.u_textures.value.distance_map = this.computes.distanceMap.texture
        this.material.defines.SKIPPING_METHOD = Configs.SkippingMethods.findIndex((x) => x === this.configs.skippingMethod)
        this.material.needsUpdate = true
    }

    onChangeGradientsMethod(event)
    {
        this.material.defines.GRADIENTS_METHOD = Configs.GradientsMethods.findIndex((x) => x === this.configs.gradientsMethod)
        this.material.needsUpdate = true
    }

    onChangeMarchingMethod(event)
    {
        this.material.defines.MARCHING_METHOD = Configs.MarchingMethods.findIndex((x) => x === this.configs.marchingMethod)
        this.material.needsUpdate = true
    }

    onChangeSkippingEnabled(event)
    {
        this.material.defines.SKIPPING_ENABLED = Number(this.configs.skippingEnabled)
        this.material.needsUpdate = true
    }

    onChangeColormap(event)
    {
        this.material.uniforms.u_shading.value.colormap = Configs.Colormaps.findIndex((x) => x === this.configs.colormap)
    }

    destroy() 
    {
        if (this.mesh) 
        {
            this.mesh.geometry.dispose()
            this.mesh.material.dispose()
            this.mesh = null
        }

        // Clean up references
        this.scene = null
        this.resources = null
        this.renderer = null
        this.camera = null
        this.sizes = null
        this.debug = null

        console.log("ISOViewer destroyed")
    } 
}