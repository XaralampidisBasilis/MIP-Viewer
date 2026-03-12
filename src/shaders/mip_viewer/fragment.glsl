precision highp float;
precision highp int;

in vec3 v_position;
in vec3 v_camera_position;
in vec3 v_camera_direction;

flat in vec3  v_ray_direction;
flat in vec3  v_ray_inv_direction;
flat in float v_ray_spacing;
flat in ivec3 v_ray_signs;
flat in uint  v_ray_axis;
flat in uint  v_ray_idx;
flat in uint  v_ray_map;

out vec4 fragColor;

#include "./chunks/utils/utils"
#include "./chunks/constants/constants"
#include "./chunks/uniforms/uniforms"
#include "./chunks/structs/structs"
#include "./chunks/functions/functions"

void main() 
{
    #include "./chunks/structs/set_structs"
    #include "./chunks/raycast/compute_raycast"
    #include "./chunks/march/compute_march"
    #include "./chunks/shade/compute_shade"

    #if DEBUG_ENABLED == 1
    #include "./chunks/debug/compute_debug"
    #endif

    #if DISCARDING_ENABLED == 1
    fragColor.a *= hit.discarded ? 0.0 : 1.0;
    #endif
}
