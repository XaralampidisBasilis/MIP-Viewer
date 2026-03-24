#ifndef STRUCT_CELL
#define STRUCT_CELL

struct Cell 
{
    bool  terminated;
    ivec3 coords;
    float entry_distance;
    float exit_distance;
    float span_distance;
    vec3  entry_position;
    vec3  exit_position;
    vec3  min_position;
    vec3  max_position;
    ivec3 entry_step;
    ivec3 exit_step;
};

Cell cell; // Global mutable struct

void set_cell()
{
    cell.terminated     = false;
    cell.coords         = ivec3(0);
    cell.entry_distance = 0.0;
    cell.exit_distance  = 0.0;
    cell.span_distance  = 0.0;
    cell.entry_position = vec3(0.0);
    cell.exit_position  = vec3(0.0);
    cell.min_position   = vec3(0.0);
    cell.max_position   = vec3(0.0);
    cell.exit_step      = ivec3(0);
    cell.entry_step     = ivec3(0);
}

#endif 
