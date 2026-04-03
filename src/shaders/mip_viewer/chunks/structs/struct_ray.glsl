#ifndef STRUCT_RAY
#define STRUCT_RAY

struct Ray 
{
    bool  discarded;       // flag indicating if the ray has been discarded
    float phase;     
    vec3  origin_position;
    vec3  direction;       // direction vector for each step along the ray
    float step_distance;         // fixed step distance for each ray 
    float eps_distance;
    vec3  eps_direction;
    float start_distance;  // starting distance along the current ray from origin for ray march  
    vec3  start_position;  // starting position of the current ray in 3d model coordinates for ray march
    float end_distance;    // ending distance along the current ray from origin for ray march
    vec3  end_position;    // ending position of the current ray in 3d model coordinates for ray march
    float span_distance;   // total distance that can be covered by the current ray for ray march
};

Ray ray; // Global mutable struct

#endif // STRUCT_RAY
