precision highp float;
precision highp int;

in vec3 v_ray_origin;
in vec3 v_vertex_position;

out vec4 fragColor;

#include "./chunks/utils/utils"
#include "./chunks/uniforms/uniforms"
#include "./chunks/structs/structs"
#include "./chunks/functions/functions"

vec3 getRayOriginIndexSpace()
{
    vec2 uv = (gl_FragCoord.xy + vec2(0.5)) / u_transform.resolution;
    vec2 ndc = uv * 2.0 - 1.0;

    vec4 clipPosition = vec4(ndc, -1.0, 1.0);
    vec4 viewPosition = u_transform.inv_projection * clipPosition;
    vec4 worldPosition = u_transform.inv_view * vec4(viewPosition.xyz, 1.0);
    vec4 localPosition = u_transform.inv_model * worldPosition;
    vec3 indexPosition = (localPosition.xyz + 0.5) * vec3(u_volume.dimensions);

    return indexPosition;
}

void main() 
{
    #include "./chunks/structs/set_structs"
    // ray.origin = getRayOriginIndexSpace();

    #include "./chunks/raycast/compute_raycast"
    #include "./chunks/march/compute_march"
    #include "./chunks/shade/compute_shade"

    #if DEBUG_ENABLED == 1
    #include "./chunks/debug/compute_debug"
    #endif
}
