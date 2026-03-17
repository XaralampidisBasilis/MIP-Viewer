precision highp float;
precision highp int;

in vec3 v_ray_origin;

flat in float v_box_min_distance;
flat in float v_box_max_distance;

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
}
