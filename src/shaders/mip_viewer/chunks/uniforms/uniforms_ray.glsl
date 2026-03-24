#ifndef UNIFORMS_RAY
#define UNIFORMS_RAY

struct UniformsRay
{
    vec3   direction;
    vec3   inv_direction;
    ivec3  sign_direction;
    vec3   step_distances;
    float  step_distance;
    uint   axis;
    uint   idx;
    uint   map;
    bool   reverse;
};

uniform UniformsRay u_ray;

#endif
