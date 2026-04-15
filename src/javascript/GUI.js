
import Experience from './Experience'
import Configs from './Configs'
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
            blockSize           : this.configs.blockSize,
            downscaleFactor     : this.configs.downscaleFactor,
            marchingMethod      : this.configs.marchingMethod,
            skippingStrategy    : this.configs.skippingStrategy,    
            skippingMethod      : this.configs.skippingMethod,    
            skippingEnabled     : this.configs.skippingEnabled,
        }

        this.controllers.configs = 
        {
            // blockSize : folder.add(objects, 'blockSize').min(2).max(8).step(1)
            // .onFinishChange((value) => 
            // { 
            //     this.configs.set('blockSize', value) 
            // }),

            // // DANGEROUS for WebGL Context Loss
            // downscaleFactor : folder.add(objects, 'downscaleFactor').min(0).max(1).step(0.05)
            // .onFinishChange((value) => 
            // { 
            //     this.configs.set('downscaleFactor', value) 
            // }),

            marchingMethod: folder.add(objects, 'marchingMethod').options(Configs.MarchingMethods)
            .onFinishChange((option) => 
            { 
                this.configs.set('marchingMethod', option) 
            }),

            skippingMethod: folder.add(objects, 'skippingMethod').options(Configs.SkippingMethods)
            .onFinishChange((option) => 
            { 
                this.configs.set('skippingMethod', option) 
            }),

            skippingEnabled : folder.add(objects, 'skippingEnabled')
            .onFinishChange((boolean) => 
            { 
                this.configs.set('skippingEnabled', boolean) 
            }),
        
        }
    }

    addControlsShading()
    {
        const material = this.viewer.material

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
            bernsteinEnabled: Boolean(defines.BERNSTEIN_ENABLED),
            debugEnabled     : Boolean(defines.DEBUG_ENABLED),
            productionEnabled: Boolean(defines.PRODUCTION_ENABLED),
            variationEnabled : Boolean(defines.VARIATION_ENABLED),
        }

        this.controllers.debug = 
        {
            option: folder.add(uniforms, 'option').options(
            { 
                default           : 0,
                 
                ray_discarded     : 101,
                ray_direction     : 102,
                ray_sign_direction: 103,
                ray_step_distance : 104,
                ray_start_distance: 105,
                ray_end_distance  : 106,
                ray_span_distance : 107,
                ray_start_position: 108,
                ray_end_position  : 109,
                ray_dominant_axis : 110,
                ray_quadrant_index: 111,
                ray_group_index   : 112,
                ray_reversed      : 113,
                ray_phase         : 114,

                block_empty         : 402,
                block_terminated    : 403,
                block_coords        : 404,
                block_step_radius   : 401,
                block_entry_step    : 405,
                block_exit_step     : 406,
                block_entry_distance: 407,
                block_exit_distance : 408,
                block_span_distance : 409,
                
                cell_terminated    : 202,
                cell_coords        : 204,
                cell_exit_step     : 205,
                cell_entry_distance: 206,
                cell_exit_distance : 207,
                cell_span_distance : 208,
                cell_far_distances : 209,

                trace_intersected: 301,
                trace_terminated : 302,
                trace_distance   : 303,
                trace_position   : 304,
                trace_residue    : 305,

                mip_terminated: 451,
                mip_update    : 452,
                mip_distance  : 453,
                mip_position  : 454,
                mip_value     : 455,
                mip_normal    : 456,
                mip_gradient  : 457,
                mip_steepness : 458,
                mip_curvatures: 459,
        
                frag_color_material: 511,
                frag_color         : 516,
                
                cubic_maximize        : 801,
                cubic_coeffs          : 802,
                cubic_bernstein_coeffs: 803,
                cubic_argmax_point    : 804,
                cubic_max_value       : 805,

                stats_num_cells           : 901,
                stats_num_traces          : 902,
                stats_num_mips            : 903,
                stats_num_maxima          : 904,
                stats_num_blocks          : 905,
                stats_num_groups          : 906,
                stats_num_fetches         : 907,
                stats_num_volume_fetches  : 908,
                stats_num_distance_fetches: 909,
                
                debug_variable0: 1000,
                debug_variable1: 1001,
                debug_variable2: 1002,
                debug_variable3: 1003,
                debug_variable4: 1004,
                debug_variable5: 1005,
                debug_variable6: 1006,
                debug_variable7: 1007,
                debug_variable8: 1008,
                debug_variable9: 1009,
            }),       

            debugEnabled: folder.add(objects, 'debugEnabled')
            .onFinishChange((value) => 
            { 
                defines.DEBUG_ENABLED = Number(value)
                material.needsUpdate = true 
            }),

            productionEnabled: folder.add(objects, 'productionEnabled')
            .onFinishChange((value) => 
            { 
                defines.PRODUCTION_ENABLED = Number(value)
                material.needsUpdate = true 
            }),

            bernsteinEnabled: folder.add(objects, 'bernsteinEnabled')
            .onFinishChange((value) => 
            { 
                defines.BERNSTEIN_ENABLED = Number(value)
                material.needsUpdate = true 
            }),

            variationEnabled: folder.add(objects, 'variationEnabled')
            .onFinishChange((value) => 
            { 
                defines.VARIATION_ENABLED = Number(value)
                material.needsUpdate = true 
            }),

            maxBlocks: folder.add(uniforms, 'max_blocks').min(0).max(defines.MAX_BLOCKS).step(1),
            maxCells : folder.add(uniforms, 'max_cells').min(0).max(defines.MAX_CELLS).step(1),
            maxTraces : folder.add(uniforms, 'max_traces').min(0).max(defines.MAX_TRACES).step(1),
            variable1 : folder.add(uniforms, 'variable1').min(0).max(1).step(1e-6),
            variable2 : folder.add(uniforms, 'variable2').min(0).max(1).step(1e-6),
            variable3 : folder.add(uniforms, 'variable3').min(0).max(1).step(1e-6),
            variable4 : folder.add(uniforms, 'variable4').min(0).max(1).step(1e-6),
            variable5 : folder.add(uniforms, 'variable5').min(0).max(1).step(1e-6),
        }
    }
    
    // controllers bindings

    destroy() {
        this.instance?.destroy()

        this.controllers = null
        this.folders = null
        this.experience = null
        this.configs = null
        this.viewer = null
        this.instance = null
    }
    
}
