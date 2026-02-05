#ifndef STRUCT_MIP
#define STRUCT_MIP

// struct to hold information about the maximum intensity projected trace
struct Mip 
{
    vec3  position;           
    float distance;   
    float value;        
    vec3  gradient;   
    mat3  hessian;   
    vec3  normal;   
    vec2  curvatures;    
};

Mip mip; // Global mutable struct

void set_mip()
{
    mip.position    = vec3(0.0);
    mip.distance    = 0.0;
    mip.value       = 0.0;
    mip.gradient    = vec3(0.0);
    mip.hessian     = mat3(0.0);
    mip.normal      = vec3(0.0);
    mip.curvatures  = vec2(0.0);
}

#endif // STRUCT_MIP
