#ifndef STRUCT_FRAG
#define STRUCT_FRAG

struct Frag 
{
    float depth;             // depth traveled from camera in NDC space
    vec3  position;          // position in NDC space
    vec3  color_material;      // color mapped from the voxel value
    vec3  color_ambient;
    vec3  color_diffuse;
    vec3  color_specular;
    vec3  color_directional;
    vec3  color;           // color after shading has been applied
};

Frag frag; // Global mutable struct

#endif // STRUCT_FRAG
