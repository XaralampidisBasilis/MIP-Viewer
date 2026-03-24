

// COMPUTE DEBUG 

// skip distance
vec4 debug_block_step_radius = to_color(turboBurn(float(block.step_radius) / 31.0));

// empty
vec4 debug_block_empty = to_color(block.empty);

// terminated
vec4 debug_block_terminated = to_color(block.terminated);

// coords
vec4 debug_block_coords = to_color(vec3(block.coords) / vec3(textureSize(u_textures.distance_map, 0) - 1));

// entry_step
vec4 debug_block_entry_step = to_color(vec3(block.entry_step));

// exit_step
vec4 debug_block_exit_step = to_color(vec3(block.exit_step));

// entry distance
vec4 debug_block_entry_distance = to_color(map(box.min_distance, box.max_distance, block.entry_distance));

// exit distance
vec4 debug_block_exit_distance = to_color(map(box.min_distance, box.max_distance, block.exit_distance));

// span distance
vec4 debug_block_span_distance = to_color(block.span_distance / (float(block.step_radius * BLOCK_SIZE) * sqrt(3.0))); 

// PRINT DEBUG
switch (u_debug.option - 400)
{
    case  1: fragColor = debug_block_step_radius;    break;
    case  2: fragColor = debug_block_empty;       break;
    case  3: fragColor = debug_block_terminated;     break;
    case  4: fragColor = debug_block_coords;         break;
    case  5: fragColor = debug_block_entry_step;     break;
    case  6: fragColor = debug_block_exit_step;      break;
    case  7: fragColor = debug_block_entry_distance; break;
    case  8: fragColor = debug_block_exit_distance;  break;
    case  9: fragColor = debug_block_span_distance;  break;
}

  