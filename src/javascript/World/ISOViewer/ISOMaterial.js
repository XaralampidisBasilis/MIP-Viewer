import * as THREE from 'three'
import vertexShader from '../../../shaders/iso_viewer/vertex.glsl'
import fragmentShader from '../../../shaders/iso_viewer/fragment.glsl'

// console.log('vertexShader: ', vertexShader)
// console.log('fragmentShader: ', fragmentShader)

export default function()
{
    const uniforms = 
    {
        uCustomModelMatrix: new THREE.Uniform
        (
            new THREE.Matrix4()
        ),

        u_volume: new THREE.Uniform
        ({
            isovalue          : 0.69,
            dimensions        : new THREE.Vector3(),
            spacing           : new THREE.Vector3(),
            spacing_normalized: new THREE.Vector3(),
            inv_dimensions    : new THREE.Vector3(),
            blocked_dimensions: new THREE.Vector3(),
            block_size        : 0,
        }),

        u_bbox: new THREE.Uniform
        ({
            min_position: new THREE.Vector3(),
            max_position: new THREE.Vector3(),
        }),

        u_textures: new THREE.Uniform
        ({
            interpolation_map : null,
            occupancy_map : null,
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
            variable1 : 0,
            variable2 : 0,
            variable3 : 0,
            variable4 : 0,
            variable5 : 0,
        }),
    }

    const defines = 
    {           
    
        MARCHING_METHOD     : 1,
        INTERPOLATION_METHOD: 2,
        SKIPPING_STRATEGY   : 2,
        SKIPPING_METHOD     : 2,
        GRADIENTS_METHOD    : 3,
        INTERSECTION_TEST   : 0,

        BERNSTEIN_ENABLED: 1,
        SKIPPING_ENABLED : 1,
        BBOX_ENABLED     : 1,

        DEBUG_ENABLED     : 1,
        DISCARDING_ENABLED: 1,
        VARIATION_ENABLED : 0,
        VARIATION_METHOD  : 1,

        MAX_CELLS           : 1000,
        MAX_TRACES          : 1000 * 5,
        MAX_BLOCKS          : 1000,
        MAX_GROUPS          : 100,
        MAX_CELLS_IN_BLOCK  : 10,
        MAX_TRACES_IN_BLOCK : 50,
        MAX_BLOCKS_IN_GROUP : 20,
    }

    const material = new THREE.ShaderMaterial
    ({    
        side: THREE.BackSide,
        blending: THREE.NormalBlending,
        depthTest: false,
        depthWrite: false,
        transparent: true,           

        glslVersion: THREE.GLSL3,
        uniforms: uniforms,
        defines: defines,
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
    })

    return material
}