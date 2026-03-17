#ifndef STRUCT_BOX
#define STRUCT_BOX

struct Box 
{  
    vec3  min_position;     
    vec3  max_position;   
    float min_distance;
    float max_distance;     
    float span_distance;   
   
};

Box box; // Global mutable struct

void set_box()
{
    box.min_position  = vec3(0.0);
    box.max_position  = vec3(u_volume.dimensions);
    box.min_distance  = v_box_min_distance;
    box.max_distance  = v_box_max_distance;
    box.span_distance = v_box_max_distance - v_box_min_distance;
}

#endif // STRUCT_BOX
