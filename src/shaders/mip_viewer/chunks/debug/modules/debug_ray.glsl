// COMPUTE DEBUG 

// discarded
vec4 debug_ray_discarded = to_color(ray.discarded);

// direction
vec4 debug_ray_direction = to_color(ray.direction * 0.5 + 0.5);

// signs
vec4 debug_ray_sign_direction = to_color(vec3(u_ray.sign_direction) * 0.5 + 0.5);

// spacing
vec4 debug_ray_step_distance = to_color(ray.step_distance);

// start distance
vec4 debug_ray_start_distance = to_color(map(u_box.min_distance, u_box.max_distance, ray.start_distance));

// end distance
vec4 debug_ray_end_distance = to_color(map(u_box.min_distance, u_box.max_distance, ray.end_distance));

// span distance
vec4 debug_ray_span_distance = to_color(map(0.0, u_box.span_distance, ray.span_distance));

// start position
vec4 debug_ray_start_position = to_color(map(u_box.min_position, u_box.max_position, ray.start_position));

// end position
vec4 debug_ray_end_position = to_color(map(u_box.min_position, u_box.max_position, ray.end_position));

// dominant axis
vec4 debug_ray_dominant_axis = to_color(vec3[3](COLOR.RED, COLOR.GREEN, COLOR.BLUE)[u_ray.dominant_axis]);

// quadrant index
vec4 debug_ray_quadrant_index = to_color(vec3[4](1.00 * COLOR.WHITE, 0.75 * COLOR.WHITE, 0.50 * COLOR.WHITE, 0.25 * COLOR.WHITE)[u_ray.quadrant_index]);

// group index
vec4 debug_ray_group_index = to_color(vec3[12](
    1.00 * COLOR.RED,   0.75 * COLOR.RED,    0.50 * COLOR.RED,   0.25 * COLOR.RED,   
    1.00 * COLOR.GREEN, 0.75 * COLOR.GREEN,  0.50 * COLOR.GREEN, 0.25 * COLOR.GREEN, 
    1.00 * COLOR.BLUE,  0.75 * COLOR.BLUE,   0.50 * COLOR.BLUE,  0.25 * COLOR.BLUE
)[u_ray.group_index]);

// inverted
vec4 debug_ray_reversed = to_color(ray.reversed);

// phase
vec4 debug_ray_phase = to_color(ray.phase);


// PRINT DEBUG
switch (u_debug.option - 100)
{
    case  1: fragColor = debug_ray_discarded;       break;
    case  2: fragColor = debug_ray_direction;       break;
    case  3: fragColor = debug_ray_sign_direction;  break;
    case  4: fragColor = debug_ray_step_distance;   break;
    case  5: fragColor = debug_ray_start_distance;  break;
    case  6: fragColor = debug_ray_end_distance;    break;
    case  7: fragColor = debug_ray_span_distance;   break;
    case  8: fragColor = debug_ray_start_position;  break;
    case  9: fragColor = debug_ray_end_position;    break;
    case 10: fragColor = debug_ray_dominant_axis;   break;
    case 11: fragColor = debug_ray_quadrant_index;  break;
    case 12: fragColor = debug_ray_group_index;     break;
    case 13: fragColor = debug_ray_reversed;        break;
    case 14: fragColor = debug_ray_phase;           break;
}