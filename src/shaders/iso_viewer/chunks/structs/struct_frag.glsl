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

void set_frag()
{
    frag.depth             = 0.0;
    frag.position          = vec3(0.0);
    frag.color_material    = vec3(0.0);
    frag.color_ambient     = vec3(0.0);
    frag.color_diffuse     = vec3(0.0);
    frag.color_specular    = vec3(0.0);
    frag.color_directional = vec3(0.0);
    frag.color             = vec3(0.0);
}

#endif // STRUCT_FRAG
