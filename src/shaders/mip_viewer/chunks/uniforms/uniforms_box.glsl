#ifndef UNIFORMS_BOX
#define UNIFORMS_BOX

struct UniformsBox
{
    vec3   min_position;
    vec3   max_position;
    float  min_distance;
    float  max_distance;
    float  span_distance;
};

uniform UniformsBox u_box;

#endif
