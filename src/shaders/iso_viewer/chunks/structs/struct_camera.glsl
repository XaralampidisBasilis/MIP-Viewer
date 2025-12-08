#ifndef STRUCT_CAMERA
#define STRUCT_CAMERA

struct Camera 
{
    vec3  position;       // position in model coordinates 
    vec3  direction;      // normalized direction in model coordinates 
};

Camera camera; // Global mutable struct

void set_camera()
{
    camera.position = v_camera_position;
    camera.direction = normalize(v_camera_direction);
}

#endif // STRUCT_CAMERA
