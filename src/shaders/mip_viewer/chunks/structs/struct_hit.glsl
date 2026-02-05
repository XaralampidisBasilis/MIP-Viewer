#ifndef STRUCT_HIT
#define STRUCT_HIT

// struct to hold information about the current ray hit 
struct Hit 
{
    bool  discarded;          
    bool  escaped;          
    bool  undefined;
    vec3  position;           
    float distance;   
    float value;        
    float residue;
    float derivative;
    float orientation;   
    vec3  gradient;   
    mat3  hessian;   
    vec3  normal;   
    vec2  curvatures;    
};

Hit hit; // Global mutable struct

void set_hit()
{
    hit.discarded   = false;
    hit.escaped     = false;
    hit.undefined   = false;
    hit.position    = vec3(0.0);
    hit.distance    = 0.0;
    hit.value       = 0.0;
    hit.residue     = 0.0;
    hit.derivative  = 0.0;
    hit.orientation = 0.0;
    hit.gradient    = vec3(0.0);
    hit.hessian     = mat3(0.0);
    hit.normal      = vec3(0.0);
    hit.curvatures  = vec2(0.0);
}

#endif // STRUCT_HIT
