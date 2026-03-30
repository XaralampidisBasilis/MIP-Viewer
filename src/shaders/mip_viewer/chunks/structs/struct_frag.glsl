#ifndef STRUCT_FRAG
#define STRUCT_FRAG

struct Frag 
{
    float depth;             // depth traveled from camera in NDC space
    vec3  position;          // position in NDC space
    vec3  color;           // color after shading has been applied
};

Frag frag; // Global mutable struct

#endif // STRUCT_FRAG
