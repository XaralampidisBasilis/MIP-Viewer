#ifndef STRUCT_MIP
#define STRUCT_MIP

// struct to hold information about the maximum intensity projected trace
struct Mip 
{
    bool  update;           
    bool  intersected;
    bool  terminated;           
    vec3  position;           
    float distance;   
    float value;        
    vec3  gradient;   
    mat3  hessian;   
    vec3  normal;   
    vec2  curvatures;    
};

Mip mip; // Global mutable struct

#endif // STRUCT_MIP
