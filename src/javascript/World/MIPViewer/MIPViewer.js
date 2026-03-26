import * as THREE from 'three'
import Experience from '../../Experience'
import EventEmitter from '../../Utils/EventEmitter'
import Configs from '../../Configs'
import ISOMaterial from './MIPMaterial'
import { updateRayUniforms } from './RayUniforms'

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
        this.camera = this.experience.camera
        this.onCameraChange = this.onCameraChange.bind(this)
        this.cameraBound = false

        this.setMesh()
    }

    start()
    {
        this.setMaterial()
        this.size = this.computes.volumeMap.size

        // scaled the unit cube into physical/world size, using the volume actual size.
        this.mesh.scale.copy(this.size) 
        
        this.bindCamera()
        this.updateRayUniforms()
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
        this.setDefinesEnablers()
        this.setDefinesMethods()
        this.setDefinesIterators()
        this.setUniformsTextures()
        this.setUniformsVolume()
        this.setUniformsShading()
        this.updateRayUniforms()
    }

    bindCamera()
    {
        if (this.cameraBound)
        {
            return
        }

        this.camera.on('change.mipViewer', this.onCameraChange)
        this.cameraBound = true
    }

    onCameraChange()
    {
        this.updateRayUniforms()
    }

    updateRayUniforms()
    {
        updateRayUniforms(
            this.material.uniforms,
            this.camera.instance,
            this.mesh,
            this.computes.volumeMap.dimensions,
        )
    }

    setDefinesEnablers()
    {
        const configs = this.configs
        const defines = this.material.defines
        defines.SKIPPING_ENABLED = Number(configs.skippingEnabled)
        defines.DEBUG_ENABLED = Number(configs.debugEnabled)
        this.material.needsUpdate = true
    }

    setDefinesMethods()
    {
        const configs = this.configs
        const defines = this.material.defines
        defines.MARCHING_METHOD = Configs.MarchingMethods.findIndex((x) => x === configs.marchingMethod)
        defines.SKIPPING_METHOD = Configs.SkippingMethods.findIndex((x) => x === configs.skippingMethod)
        defines.DISTANCE_VARIATION = Configs.DistanceVariations.findIndex((x) => x === configs.distanceVariation)
        this.material.needsUpdate = true
    }

    setDefinesIterators()
    {        
        const defines = this.material.defines

        defines.BLOCK_SIZE = this.configs.blockSize
        defines.MAX_BLOCKS = this.computes.distanceMap.dimensions.toArray().reduce((y, x) => y + x, -2)
        
        defines.MAX_CELLS = this.computes.volumeMap.dimensions.toArray().reduce((y, x) => y + x, -2)
        defines.MAX_CELLS_IN_BLOCK = defines.BLOCK_SIZE * 3 - 2

        defines.MAX_TRACES_IN_CELL = 4
        defines.MAX_TRACES = defines.MAX_TRACES_IN_CELL * defines.MAX_CELLS
        defines.MAX_TRACES_IN_BLOCK = defines.MAX_TRACES_IN_CELL * defines.MAX_CELLS_IN_BLOCK

        this.material.needsUpdate = true

        console.log(defines)
    }

    setUniformsTextures()
    {
        const uniforms = this.material.uniforms
        uniforms.u_textures.value.volume_map = this.computes.volumeMap.texture
        uniforms.u_textures.value.distance_map = this.computes.distanceMap.texture
    }

    setUniformsVolume()
    {
        const uniforms = this.material.uniforms
        uniforms.u_volume.value.dimensions.copy(this.computes.volumeMap.dimensions)
        uniforms.u_volume.value.spacing.copy(this.computes.volumeMap.spacing)
        uniforms.u_volume.value.spacing_normalized.copy(this.computes.volumeMap.spacing).normalize()
        uniforms.u_volume.value.block_size = this.configs.blockSize
        uniforms.u_volume.value.inv_dimensions.fromArray(uniforms.u_volume.value.dimensions.toArray().map(x => 1/x))
    }

    setUniformsShading()
    {
        const uniforms = this.material.uniforms
        uniforms.u_shading.value.colormap = Configs.Colormaps.findIndex((x) => x === this.configs.colormap)
    }

    change(event)
    {
        if      (event.key === 'blockSize'          ) this.onChangeBlockSize(event)
        else if (event.key === 'downscaleFactor'    ) this.onChangeDownscaleFactor(event)
        else if (event.key === 'skippingMethod'     ) this.onChangeSkippingMethod(event)
        else if (event.key === 'marchingMethod'     ) this.onChangeMarchingMethod(event)
        else if (event.key === 'skippingEnabled'    ) this.onChangeSkippingEnabled(event)
        else if (event.key === 'colormap'           ) this.onChangeColormap(event)
        
        console.log(this)
    }

    onChangeBlockSize(event)
    {
        const uniforms = this.material.uniforms
        uniforms.u_volume.value.block_size = this.configs.blockSize
        uniforms.u_volume.value.blocked_dimensions.copy(this.computes.distanceMap.dimensions)
        uniforms.u_textures.value.shadow_map.dispose()
        uniforms.u_textures.value.shadow_map = this.computes.distanceMap.texture    
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

    onChangeSkippingMethod(event)
    {
        this.material.uniforms.u_textures.value.distance_map = this.computes.distanceMap.texture
        this.material.defines.SKIPPING_METHOD = Configs.SkippingMethods.findIndex((x) => x === this.configs.skippingMethod)
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
        if (this.cameraBound)
        {
            this.camera?.off('change.mipViewer')
            this.cameraBound = false
        }

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
