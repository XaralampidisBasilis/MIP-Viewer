#ifndef STRUCT_CELL
#define STRUCT_CELL


struct Cell 
{
    bool  shadowed;
    bool  terminated;
    ivec3 coords;
    int   step_radius;
    ivec3 exit_step;
    float entry_distance;
    float exit_distance;
    float span_distance;
    vec3  entry_position;
    vec3  exit_position;
};

Cell cell; // Global mutable struct

void set_cell()
{
    cell.shadowed       = false;
    cell.terminated     = false;
    cell.coords         = ivec3(0);
    cell.step_radius  = 0;
    cell.exit_step    = ivec3(0);
    cell.entry_distance = 0.0;
    cell.exit_distance  = 0.0;
    cell.span_distance  = 0.0;
    cell.entry_position = vec3(0.0);
    cell.exit_position  = vec3(0.0);
}

#endif 
