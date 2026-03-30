#ifndef UNIFORMS_TRANSFORM
#define UNIFORMS_TRANSFORM

struct UniformsTransform    
{
    vec2   resolution;   
    mat4   inv_projection;   
    mat4   inv_view;   
    mat4   inv_model;   
};

uniform UniformsTransform u_transform;

#endif
