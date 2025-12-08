
// COMPUTE DEBUG

// terminated
vec4 debug_cell_intersected = to_color(cell.intersected);

// terminated
vec4 debug_cell_terminated = to_color(cell.terminated);

// coords
vec4 debug_cell_coords = to_color(vec3(cell.coords) * u_volume.inv_dimensions);

// axes
vec4 debug_cell_exit_normal = to_color(vec3(cell.exit_normal));

// entry distance
vec4 debug_cell_entry_distance = to_color(map(box.min_entry_distance, box.max_exit_distance, cell.entry_distance)); 

// exit distance
vec4 debug_cell_exit_distance = to_color(map(box.min_entry_distance, box.max_exit_distance, cell.exit_distance)); 

// span distance
vec4 debug_cell_span_distance = to_color(cell.span_distance / length(cell.max_position - cell.min_position)); 

// min position
vec4 debug_cell_min_position = to_color(map(box.min_position, box.max_position, cell.min_position)); 

// max position
vec4 debug_cell_max_position = to_color(map(box.min_position, box.max_position, cell.max_position)); 

// PRINT DEBUG
switch (u_debug.option - 200)
{ 
    case 1: fragColor = debug_cell_intersected;        break;
    case 2: fragColor = debug_cell_terminated;         break;
    case 3: fragColor = debug_cell_coords;             break;
    case 4: fragColor = debug_cell_exit_normal;        break;
    case 5: fragColor = debug_cell_max_position;       break;
    case 6: fragColor = debug_cell_min_position;       break;
    case 7: fragColor = debug_cell_entry_distance;     break;
    case 8: fragColor = debug_cell_exit_distance;      break;
    case 9: fragColor = debug_cell_span_distance;      break;
}