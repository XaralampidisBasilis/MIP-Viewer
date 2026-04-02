#ifndef UNIFORMS_RAY
#define UNIFORMS_RAY

struct UniformsRay
{
    float min_value;
    float max_value;
    vec3  direction;
    vec3  inv_direction;
    ivec3 sign_direction;
    vec3  step_distances;
    float step_distance;
    uint  dominant_axis;
    uint  quadrant_index;
    uint  group_index;
    bool  reverse;
};

uniform UniformsRay u_ray;

#endif
