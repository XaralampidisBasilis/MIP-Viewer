#ifndef STRUCT_TRACE
#define STRUCT_TRACE

// struct to hold information about the current ray trace 
struct Trace 
{
    bool  terminated;           // flag indicating if the trace has reached out of u_intensity_map bounds
    vec3  position;             // current position in 3d model coordinates
    float distance;             // current distance traveled from camera
    float spacing;             // current distance traveled from camera
    float value;           
    float prev_distance;           
    float mip_distance;           
    float mip_value;         
};

Trace trace; // Global mutable struct

void set_trace()
{
    trace.terminated    = false;
    trace.position      = vec3(0.0);
    trace.distance      = 0.0;
    trace.spacing       = 0.0;
    trace.value         = 0.0;
    trace.prev_distance = 0.0;
    trace.mip_distance  = 0.0;
    trace.mip_value     = 0.0;
}

#endif // STRUCT_TRACE
