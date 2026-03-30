#ifndef STRUCT_CELL
#define STRUCT_CELL

struct Cell 
{
    bool  terminated;
    ivec3 coords;
    float entry_distance;
    float exit_distance;
    vec3  far_distances;
    float span_distance;
    vec3  entry_position;
    vec3  exit_position;
    ivec3 entry_step;
    ivec3 exit_step;
};

Cell cell; // Global mutable struct

#endif 
