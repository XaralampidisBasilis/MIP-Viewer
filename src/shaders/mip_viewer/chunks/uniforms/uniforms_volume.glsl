#ifndef UNIFORMS_VOLUME
#define UNIFORMS_VOLUME

struct UniformsVolume 
{
    ivec3 dimensions;    
    vec3  spacing;           
    vec3  spacing_normalized;           
    vec3  inv_dimensions;   
    int   block_size;
};

uniform UniformsVolume u_volume;

#endif
