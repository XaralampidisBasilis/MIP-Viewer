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

        u_textures: new THREE.Uniform
        ({
            volume_map : null,
            distance_map  : null,
        }),

        u_shading: new THREE.Uniform
        ({
            colormap          : 0,
            shininess         : 40.0,
            reflect_ambient   : 0.2,
            reflect_diffuse   : 1.0,
            reflect_specular  : 0.2,
            modulate_edges    : 1.0,
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
        BLOCK_SIZE       : 1,
        MARCHING_METHOD  : 1,
        SKIPPING_METHOD  : 2,

        BERNSTEIN_ENABLED : 1,
        SKIPPING_ENABLED  : 1,
        PRODUCTION_ENABLED: 0,
        DEBUG_ENABLED     : 1,
        VARIATION_ENABLED : 0,

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
