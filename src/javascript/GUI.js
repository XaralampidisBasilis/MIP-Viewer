
import Experience from './Experience'
import Configs from './Utils/Configs'
import Gui from 'lil-gui'

export default class GUI
{
    constructor()
    {
        this.experience = new Experience()
        this.configs = this.experience.configs
        this.viewer = this.experience.world.viewer
        this.instance = new Gui() 
        this.instance.close()
    }

    start()
    {
        this.addFolders()
        this.addControls()
        // this.addToggles()
    }

    addFolders()
    {        
        this.folders = {}
        this.folders.configs = this.instance.addFolder('Configs').close()
        this.folders.shading = this.instance.addFolder('Shading').close()
        this.folders.debug = this.instance.addFolder('Debug').close()
    }

    addToggles()
    {
        const folders = Object.values(this.folders)

        const closeOtherFolders = (openFolder) => 
        {
            folders.forEach((folder) => 
            {
                if (folder !== openFolder && !folder._closed) folder.close()
            })
        }

        folders.forEach((folder) => 
        {
            folder.onOpenClose((openFolder) => 
            {
                if (!openFolder._closed) closeOtherFolders(openFolder)
            })
        })
    }

    // controllers
    
    addControls()
    {
        this.controllers = {}
        this.addControlsConfigs() 
        this.addControlsShading()
        this.addControlsDebug() 
    }

    addControlsConfigs() 
    {
        const folder = this.folders.configs
        const objects = 
        { 
            isosurfaceValue     : this.configs.isosurfaceValue,
            blockSize           : this.configs.blockSize,
            downscaleFactor     : this.configs.downscaleFactor,
            interpolationMethod : this.configs.interpolationMethod,
            intersectionTest    : this.configs.intersectionTest,
            gradientsMethod     : this.configs.gradientsMethod,
            marchingMethod      : this.configs.marchingMethod,
            skippingStrategy    : this.configs.skippingStrategy,    
            skippingMethod      : this.configs.skippingMethod,    
            bernsteinEnabled    : this.configs.bernsteinEnabled,
            skippingEnabled     : this.configs.skippingEnabled,  
            boundingBoxEnabled  : this.configs.boundingBoxEnabled,  
        }
    
        this.controllers.configs = 
        {
            isosurfaceValue: folder.add(objects, 'isosurfaceValue').min(0).max(1).step(0.0001)
            .onFinishChange((value) => 
            { 
                this.configs.set('isosurfaceValue', value) 
            }),

            // blockSize : folder.add(objects, 'blockSize').min(2).max(8).step(1)
            // .onFinishChange((value) => 
            // { 
            //     this.configs.set('blockSize', value) 
            // }),

            // // DANGEROUS for WebGL Context Loss
            // downscaleFactor : folder.add(objects, 'downscaleFactor').min(0).max(0.8).step(0.1)
            // .onFinishChange((value) => 
            // { 
            //     this.configs.set('downscaleFactor', value) 
            // }),
            
            marchingMethod: folder.add(objects, 'marchingMethod').options(Configs.MarchingMethods)
            .onFinishChange((option) => 
            { 
                this.configs.set('marchingMethod', option) 
            }),

            interpolationMethod: folder.add(objects, 'interpolationMethod').options(Configs.InterpolationMethods)
            .onFinishChange((option) => 
            { 
                this.configs.set('interpolationMethod', option) 
            }),

            // skippingStrategy: folder.add(objects, 'skippingStrategy').options(Configs.SkippingStrategies)
            // .onFinishChange((option) => 
            // { 
            //     this.configs.set('skippingStrategy', option) 
            // }),

            skippingMethod: folder.add(objects, 'skippingMethod').options(Configs.SkippingMethods)
            .onFinishChange((option) => 
            { 
                this.configs.set('skippingMethod', option) 
            }),

            // intersectionTest: folder.add(objects, 'intersectionTest').options(Configs.IntersectionTests)
            // .onFinishChange((option) => 
            // { 
            //     this.configs.set('intersectionTest', option) 
            // }),

            gradientsMethod: folder.add(objects, 'gradientsMethod').options(Configs.GradientsMethods)
            .onFinishChange((option) => 
            { 
                this.configs.set('gradientsMethod', option) 
            }),

            bernsteinEnabled : folder.add(objects, 'bernsteinEnabled')
            .onFinishChange((boolean) => 
            { 
                this.configs.set('bernsteinEnabled', boolean) 
            }),

            skippingEnabled : folder.add(objects, 'skippingEnabled')
            .onFinishChange((boolean) => 
            { 
                this.configs.set('skippingEnabled', boolean) 
            }),
            
            boundingBoxEnabled : folder.add(objects, 'boundingBoxEnabled')
            .onFinishChange((boolean) => 
            { 
                this.configs.set('boundingBoxEnabled', boolean) 
            }),
        }
    }

    addControlsShading()
    {
        const material = this.viewer.material
        const uniforms = material.uniforms.u_shading.value

        const folder = this.folders.shading
        const objects = 
        {
            colormap : this.configs.colormap,
        }

        this.controllers.shading =
        {
            colormap : folder.add(objects, 'colormap').options(Configs.Colormaps)
            .onFinishChange((option) => 
            { 
                this.configs.set('colormap', option) 
            }),

            shininess        : folder.add(uniforms, 'shininess').min(0).max(60.0).step(0.1),
            reflectAmbient   : folder.add(uniforms, 'reflect_ambient').min(0).max(1).step(0.001),
            reflectDiffuse   : folder.add(uniforms, 'reflect_diffuse').min(0).max(1).step(0.001),
            reflectSpecular  : folder.add(uniforms, 'reflect_specular').min(0).max(1).step(0.001),
            modulateEdges    : folder.add(uniforms, 'modulate_edges').min(0).max(1).step(0.001),
            modulateGradient : folder.add(uniforms, 'modulate_gradient').min(0).max(1).step(0.001),
            modulateCurvature: folder.add(uniforms, 'modulate_curvature').min(0).max(1).step(0.001),
        } 
    }
    
    addControlsDebug()
    {
        const material = this.viewer.material
        const uniforms = material.uniforms.u_debug.value
        const defines = material.defines

        const folder = this.folders.debug
        const objects = 
        { 
            discardingEnabled: Boolean(defines.DISCARDING_ENABLED),
            debugEnabled     : Boolean(defines.DEBUG_ENABLED),
            variationEnabled : Boolean(defines.VARIATION_ENABLED),
            variationMethod  : Number(defines.VARIATION_METHOD),
        }

        this.controllers.debug = 
        {
            option: folder.add(uniforms, 'option').options(
            { 
                default                 : 0,
                 
                ray_discarded           : 101,
                ray_direction           : 102,
                ray_signs               : 103,
                ray_spacing             : 104,
                ray_start_distance      : 105,
                ray_end_distance        : 106,
                ray_span_distance       : 107,
                ray_start_position      : 108,
                ray_end_position        : 109,

                block_occupied          : 402,
                block_terminated        : 403,
                block_coords            : 404,
                block_skip_coords       : 401,
                block_exit_normal       : 405,
                block_entry_distance    : 406,
                block_exit_distance     : 407,
                block_span_distance     : 408,
                block_min_position      : 409,
                block_max_position      : 410,
                
                cell_intersected        : 201,
                cell_terminated         : 202,
                cell_coords             : 203,
                cell_exit_normal        : 204,
                cell_max_position       : 205,
                cell_min_position       : 206,
                cell_entry_distance     : 207,
                cell_exit_distance      : 208,
                cell_span_distance      : 209,

                trace_intersected       : 301,
                trace_terminated        : 302,
                trace_distance          : 303,
                trace_position          : 304,
                trace_residue           : 305,

                hit_discarded           : 451,
                hit_escaped             : 452,
                hit_undefined           : 453,
                hit_distance            : 454,
                hit_position            : 455,
                hit_residue             : 456,
                hit_derivative          : 457,
                hit_orientation         : 458,
                hit_normal              : 459,
                hit_gradient            : 460,
                hit_steepness           : 461,
                hit_curvatures          : 462,
        
                frag_color_material     : 511,
                frag_color_ambient      : 512,
                frag_color_diffuse      : 513,
                frag_color_specular     : 514,
                frag_color_directional  : 515,
                frag_color              : 516,
                frag_luminance          : 517,

                box_entry_distance: 601,
                box_exit_distance:  602,
                box_span_distance:  603,
                box_entry_position: 604,
                box_exit_position:  605,

                camera_position         : 701,
                camera_direction        : 702,

                cubic_root:              801,
                cubic_num_roots:         802,
                cubic_degree:            803,
                cubic_weights:           804,
                cubic_bernstein_weights: 805,
                cubic_bernstein_spread:  806,

                quintic_root:              851,
                quintic_num_roots:         852,
                quintic_degree:            853,
                quintic_weights:           854,
                quintic_bernstein_weights: 855,
                quintic_bernstein_spread:  856,

                stats_num_cells             : 901,
                stats_num_traces            : 902,
                stats_num_blocks            : 903,
                stats_num_groups            : 904,
                stats_num_texture_fetches   : 905,
                stats_num_intersection_tests: 906,
                
                debug_variable0         : 1000,
                debug_variable1         : 1001,
                debug_variable2         : 1002,
                debug_variable3         : 1003,
                debug_variable4         : 1004,
                debug_variable5         : 1005,
                debug_variable6         : 1006,
                debug_variable7         : 1007,
                debug_variable8         : 1008,
                debug_variable9         : 1009,
            }),       

            debugEnabled: folder.add(objects, 'debugEnabled')
            .onFinishChange((value) => 
            { 
                defines.DEBUG_ENABLED = Number(value)
                material.needsUpdate = true 
            }),

            discardingEnabled: folder.add(objects, 'discardingEnabled')
            .onFinishChange((value) => 
            { 
                defines.DISCARDING_ENABLED = Number(value)
                material.needsUpdate = true 
            }),

            variationEnabled: folder.add(objects, 'variationEnabled')
            .onFinishChange((value) => 
            { 
                defines.VARIATION_ENABLED = Number(value)
                material.needsUpdate = true 
            }),
            
            variationMethod: folder.add(objects, 'variationMethod').options([1, 2, 3, 4, 5])
            .onFinishChange((value) => 
            { 
                defines.VARIATION_METHOD = Number(value)
                material.needsUpdate = true 
            }),

            // maxGroups: folder.add(uniforms, 'max_groups').min(0).max(defines.MAX_GROUPS).step(1),
            // maxBlocks: folder.add(uniforms, 'max_blocks').min(0).max(defines.MAX_BLOCKS_IN_GROUP).step(1),
            // maxCells : folder.add(uniforms, 'max_cells').min(0).max(defines.MAX_CELLS_IN_BLOCK).step(1),
            variable1 : folder.add(uniforms, 'variable1').min(0).max(1).step(1e-6),
            variable2 : folder.add(uniforms, 'variable2').min(0).max(1).step(1e-6),
            variable3 : folder.add(uniforms, 'variable3').min(0).max(1).step(1e-6),
            variable4 : folder.add(uniforms, 'variable4').min(0).max(1).step(1e-6),
            variable5 : folder.add(uniforms, 'variable5').min(0).max(1).step(1e-6),
        }
    }
    
    // controllers bindings

    destroy() {

        // Dispose of controllers
        Object.values(this.controllers).forEach(group => 
        {
            Object.values(group).forEach(controller => 
            {
                controller.remove()
            })
        })
    
        // Dispose of folders
        Object.values(this.folders).forEach(folder => 
        {
            folder.close()
            folder.destroy()
        })
    
    
        // Clear references
        this.controllers = null
        this.folders = null
        this.experience = null
    }
    
}
