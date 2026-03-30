#ifndef STRUCT_BLOCK
#define STRUCT_BLOCK

// struct to hold the current occumap parameters
struct Block
{
    bool  empty;
    bool  prev_empty;
    bool  terminated;
    ivec3 coords;  
    int   step_radius;
    float entry_distance;
    float exit_distance;
    float span_distance;
    vec3  entry_position;
    vec3  exit_position;
    ivec3 entry_step;
    ivec3 exit_step;
};

Block block; // Global mutable struct

#endif // STRUCT_BLOCK
