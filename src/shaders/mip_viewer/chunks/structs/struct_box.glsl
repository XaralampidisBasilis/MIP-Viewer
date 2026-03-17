#ifndef STRUCT_BOX
#define STRUCT_BOX

struct Box 
{  
    vec3  entry_position;   
    float entry_distance;   
    vec3  exit_position;     
    float exit_distance;     
    float span_distance;
    vec3  min_position;     
    vec3  max_position;   
    float min_entry_distance;
    float max_exit_distance;     
    float max_span_distance;   
   
};

Box box; // Global mutable struct

void set_box()
{
    box.min_position       = vec3(0.0);
    box.max_position       = vec3(u_volume.dimensions);
    box.entry_position     = vec3(0.0);
    box.exit_position      = vec3(0.0);
    box.entry_distance     = 0.0;
    box.exit_distance      = 0.0;
    box.span_distance      = 0.0;
    box.min_entry_distance = v_box_min_distance;
    box.max_exit_distance  = v_box_max_distance;
    box.max_span_distance  = v_box_max_distance - v_box_min_distance;
}

#endif // STRUCT_BOX
