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

void set_block()
{
    block.step_radius    = 0;
    block.prev_empty     = false;
    block.empty          = false;
    block.terminated     = false;
    block.coords         = ivec3(0);
    block.entry_distance = 0.0;
    block.exit_distance  = 0.0;
    block.span_distance  = 0.0;
    block.entry_position = vec3(0.0);
    block.exit_position  = vec3(0.0);
    block.entry_step     = ivec3(0);
    block.exit_step      = ivec3(0);
}

#endif // STRUCT_BLOCK