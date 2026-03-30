#ifndef STRUCT_TRACE
#define STRUCT_TRACE

// struct to hold information about the current ray trace 
struct Trace 
{
    bool  terminated;           // flag indicating if the trace has reached out of u_intensity_map bounds
    vec3  position;             // current position in 3d model coordinates
    float distance;             // current distance traveled from camera
    float step_distance;             // current distance traveled from camera
    float value;           
    float prev_distance;           
    float mip_distance;           
    float mip_value;         
};

Trace trace; // Global mutable struct

#endif // STRUCT_TRACE
