// COMPUTE DEBUG 

// discarded
vec4 debug_ray_discarded = to_color(ray.discarded);

// direction
vec4 debug_ray_direction = to_color(ray.direction * 0.5 + 0.5);

// signs
vec4 debug_ray_signs = to_color(vec3(ray.signs) * 0.5 + 0.5);

// spacing
vec4 debug_ray_spacing = to_color(ray.spacing);

// start distance
vec4 debug_ray_start_distance = to_color(map(box.min_entry_distance, box.max_exit_distance, ray.start_distance));

// end distance
vec4 debug_ray_end_distance = to_color(map(box.min_entry_distance, box.max_exit_distance, ray.end_distance));

// span distance
vec4 debug_ray_span_distance = to_color(map(0.0, box.max_span_distance, ray.span_distance));

// start position
vec4 debug_ray_start_position = to_color(map(box.min_position, box.max_position, ray.start_position));

// end position
vec4 debug_ray_end_position = to_color(map(box.min_position, box.max_position, ray.end_position));

// map
vec4 debug_ray_map = to_color(vec3[12](
    1.0 * COLOR.RED,   0.7 * COLOR.RED,    0.4 * COLOR.RED,   0.2 * COLOR.RED,   
    1.0 * COLOR.GREEN, 0.7 * COLOR.GREEN,  0.4 * COLOR.GREEN, 0.2 * COLOR.GREEN, 
    1.0 * COLOR.BLUE,  0.7 * COLOR.BLUE,   0.4 * COLOR.BLUE,  0.2 * COLOR.BLUE
)[ray.map]);

// segment

vec4 debug_ray_segment;

if (ray.axis == 0)
{   
    if (ray.signs[0] > 0)
    {
        if (ray.signs[1] > 0 && ray.signs[2] > 0) debug_ray_segment = vec4(1.00 * COLOR.RED, 1.0);
        if (ray.signs[1] > 0 && ray.signs[2] < 0) debug_ray_segment = vec4(0.75 * COLOR.RED, 1.0);
        if (ray.signs[1] < 0 && ray.signs[2] > 0) debug_ray_segment = vec4(0.50 * COLOR.RED, 1.0);
        if (ray.signs[1] < 0 && ray.signs[2] < 0) debug_ray_segment = vec4(0.25 * COLOR.RED, 1.0);
    }
    else
    {
        if (ray.signs[1] > 0 && ray.signs[2] > 0) debug_ray_segment = vec4(1.00 * COLOR.CYAN, 1.0);
        if (ray.signs[1] > 0 && ray.signs[2] < 0) debug_ray_segment = vec4(0.75 * COLOR.CYAN, 1.0);
        if (ray.signs[1] < 0 && ray.signs[2] > 0) debug_ray_segment = vec4(0.50 * COLOR.CYAN, 1.0);
        if (ray.signs[1] < 0 && ray.signs[2] < 0) debug_ray_segment = vec4(0.25 * COLOR.CYAN, 1.0);
    }
}
if (ray.axis == 1)
{   
    if (ray.signs[1] > 0)
    {
        if (ray.signs[0] > 0 && ray.signs[2] > 0) debug_ray_segment = vec4(1.00 * COLOR.GREEN, 1.0);
        if (ray.signs[0] > 0 && ray.signs[2] < 0) debug_ray_segment = vec4(0.75 * COLOR.GREEN, 1.0);
        if (ray.signs[0] < 0 && ray.signs[2] > 0) debug_ray_segment = vec4(0.50 * COLOR.GREEN, 1.0);
        if (ray.signs[0] < 0 && ray.signs[2] < 0) debug_ray_segment = vec4(0.25 * COLOR.GREEN, 1.0);
    }
    else
    {
        if (ray.signs[0] > 0 && ray.signs[2] > 0) debug_ray_segment = vec4(1.00 * COLOR.MAGENTA, 1.0);
        if (ray.signs[0] > 0 && ray.signs[2] < 0) debug_ray_segment = vec4(0.75 * COLOR.MAGENTA, 1.0);
        if (ray.signs[0] < 0 && ray.signs[2] > 0) debug_ray_segment = vec4(0.50 * COLOR.MAGENTA, 1.0);
        if (ray.signs[0] < 0 && ray.signs[2] < 0) debug_ray_segment = vec4(0.25 * COLOR.MAGENTA, 1.0);
    }
}
if (ray.axis == 2)
{   
    if (ray.signs[2] > 0)
    {
        if (ray.signs[0] > 0 && ray.signs[1] > 0) debug_ray_segment = vec4(1.00 * COLOR.BLUE, 1.0);
        if (ray.signs[0] > 0 && ray.signs[1] < 0) debug_ray_segment = vec4(0.75 * COLOR.BLUE, 1.0);
        if (ray.signs[0] < 0 && ray.signs[1] > 0) debug_ray_segment = vec4(0.50 * COLOR.BLUE, 1.0);
        if (ray.signs[0] < 0 && ray.signs[1] < 0) debug_ray_segment = vec4(0.25 * COLOR.BLUE, 1.0);
    }
    else
    {
        if (ray.signs[0] > 0 && ray.signs[1] > 0) debug_ray_segment = vec4(1.00 * COLOR.YELLOW, 1.0);
        if (ray.signs[0] > 0 && ray.signs[1] < 0) debug_ray_segment = vec4(0.75 * COLOR.YELLOW, 1.0);
        if (ray.signs[0] < 0 && ray.signs[1] > 0) debug_ray_segment = vec4(0.50 * COLOR.YELLOW, 1.0);
        if (ray.signs[0] < 0 && ray.signs[1] < 0) debug_ray_segment = vec4(0.25 * COLOR.YELLOW, 1.0);
    }
}

// inverted
vec4 debug_ray_reversed = to_color(ray.reversed);


// PRINT DEBUG
switch (u_debug.option - 100)
{
    case  1: fragColor = debug_ray_discarded;       break;
    case  2: fragColor = debug_ray_direction;       break;
    case  3: fragColor = debug_ray_signs;           break;
    case  4: fragColor = debug_ray_spacing;         break;
    case  5: fragColor = debug_ray_start_distance;  break;
    case  6: fragColor = debug_ray_end_distance;    break;
    case  7: fragColor = debug_ray_span_distance;   break;
    case  8: fragColor = debug_ray_start_position;  break;
    case  9: fragColor = debug_ray_end_position;    break;
    case 10: fragColor = debug_ray_map;             break;
    case 11: fragColor = debug_ray_segment;         break;
    case 12: fragColor = debug_ray_reversed;        break;
}