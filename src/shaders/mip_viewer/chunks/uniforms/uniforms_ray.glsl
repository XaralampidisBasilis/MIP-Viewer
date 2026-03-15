#ifndef UNIFORMS_RAY
#define UNIFORMS_RAY

struct UniformsRay
{
    vec3   direction;
    vec3   inv_direction;
    float  spacing;
    ivec3  signs;
    int    axis;
    int    idx;
    int    map;
    int    reverse;
};

uniform UniformsRay u_ray;

#endif
