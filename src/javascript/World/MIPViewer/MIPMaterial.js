import * as THREE from 'three'
import vertexShader from '../../../shaders/mip_viewer/vertex.glsl'
import fragmentShader from '../../../shaders/mip_viewer/fragment.glsl'

// console.log('vertexShader: ', vertexShader)
// console.log('fragmentShader: ', fragmentShader)

export default function()
{
    const uniforms = 
    {
        u_volume: new THREE.Uniform
        ({
            dimensions        : new THREE.Vector3(),
            spacing           : new THREE.Vector3(),
            spacing_normalized: new THREE.Vector3(),
            inv_dimensions    : new THREE.Vector3(),
            blocked_dimensions: new THREE.Vector3(),
            block_size        : 0,
        }),

        u_ray: new THREE.Uniform
        ({
            direction     : new THREE.Vector3(),
            inv_direction : new THREE.Vector3(),
            sign_direction: new THREE.Vector3(),
            step_distances: new THREE.Vector3(),
            step_distance : 1,
            axis          : 0,
            idx           : 0,
            map           : 0,
            reverse       : 0,
        }),

        u_box: new THREE.Uniform
        ({
            min_position: new THREE.Vector3(),
            max_position: new THREE.Vector3(),
            min_distance: 0.0,
            max_distance: 0.0,
            span_distance: 0.0,
        }),

        u_transform: new THREE.Uniform
        ({
            resolution: new THREE.Vector2(),   
            inv_projection: new THREE.Matrix4(),   
            inv_view: new THREE.Matrix4(),   
            inv_model: new THREE.Matrix4(),   
        }),

        u_textures: new THREE.Uniform
        ({
            volume_map : null,
            distance_map  : null,
        }),

        u_shading: new THREE.Uniform
        ({
            colormap          : 0,
            modulate_gradient : 1.0,
            modulate_curvature: 1.0,
        }),

        u_debug: new THREE.Uniform
        ({
            option    : 0,
            max_groups: 0,
            max_blocks: 0,
            max_cells : 0,
            max_traces : 0,
            variable1 : 0,
            variable2 : 0,
            variable3 : 0,
            variable4 : 0,
            variable5 : 0,
        }),
    }

    const defines = 
    {           
        DISTANCE_VARIATION: 0,
        MARCHING_METHOD   : 1,
        SKIPPING_METHOD   : 2,

        BERNSTEIN_ENABLED : 0,
        SKIPPING_ENABLED  : 1,
        PRODUCTION_ENABLED: 0,
        DEBUG_ENABLED     : 1,
        VARIATION_ENABLED : 0,

        BLOCK_SIZE  : 1,
        MAX_TRACES_IN_CELL: 1,

        MAX_CELLS           : 1000,
        MAX_TRACES          : 1000,
        MAX_BLOCKS          : 1000,
        MAX_CELLS_IN_BLOCK  : 100,
        MAX_TRACES_IN_BLOCK : 100,
    }

    const material = new THREE.ShaderMaterial
    ({    
        side: THREE.BackSide,
        blending: THREE.NoBlending,
        depthTest: false,
        depthWrite: false,
        transparent: false,           

        glslVersion: THREE.GLSL3,
        uniforms: uniforms,
        defines: defines,
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
    })

    return material
}
