#ifndef UNIFORMS_RAY
#define UNIFORMS_RAY

struct UniformsRay
{
    vec3   direction;
    vec3   inv_direction;
    vec3   step_distances;
    float  step_distance;
    ivec3  signs;
    uint   axis;
    uint   idx;
    uint   map;
    bool   reverse;
};

uniform UniformsRay u_ray;

#endif
