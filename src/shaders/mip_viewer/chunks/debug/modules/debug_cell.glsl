
// COMPUTE DEBUG

// terminated
vec4 debug_cell_terminated = to_color(cell.terminated);

// coords
vec4 debug_cell_coords = to_color(vec3(cell.coords) * u_volume.inv_dimensions);

// axes
vec4 debug_cell_exit_step = to_color(vec3(cell.exit_step));

// entry distance
vec4 debug_cell_entry_distance = to_color(map(u_box.min_distance, u_box.max_distance, cell.entry_distance)); 

// exit distance
vec4 debug_cell_exit_distance = to_color(map(u_box.min_distance, u_box.max_distance, cell.exit_distance)); 

// span distance
vec4 debug_cell_span_distance = to_color(cell.span_distance / sqrt(3.0)); 

// far distances
vec4 debug_cell_far_distances = to_color(map(u_box.min_distance, u_box.max_distance, cell.far_distances));

// PRINT DEBUG
switch (u_debug.option - 200)
{ 
    case 2: fragColor = debug_cell_terminated;     break;
    case 4: fragColor = debug_cell_coords;         break;
    case 5: fragColor = debug_cell_exit_step;      break;
    case 6: fragColor = debug_cell_entry_distance; break;
    case 7: fragColor = debug_cell_exit_distance;  break;
    case 8: fragColor = debug_cell_span_distance;  break;
    case 9: fragColor = debug_cell_far_distances;  break;
}