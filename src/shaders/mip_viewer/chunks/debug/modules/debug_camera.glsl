// COMPUTE DEBUG

// direction
vec4 debug_camera_direction = to_color(camera.direction * 0.5 + 0.5);

// position
vec4 debug_camera_position = to_color(map(box.min_position, box.max_position, camera.position));


// PRINT DEBUG
switch (u_debug.option - 700)
{
    case 1: fragColor = debug_camera_position;  break;
    case 2: fragColor = debug_camera_direction; break;
}