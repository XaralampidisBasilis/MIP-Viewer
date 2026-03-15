// compute skip distance
block.skip_distance = sample_rgba16ui_distance_fast(block.coords, block.shadowed);
// block.skip_distance = sample_rgb32ui_distance_fast(block.coords, block.shadowed);

// compute min/max coords
block.min_coords = block.coords - (block.skip_distance - 1);
block.max_coords = block.coords + (block.skip_distance - 1);

// compute min/max positions
block.min_position = vec3((block.min_coords + 0) * u_volume.block_size) - 0.5;
block.max_position = vec3((block.max_coords + 1) * u_volume.block_size) - 0.5;  

// compute inflation to avoid instabilities
block.min_position -= 0.001;
block.max_position += 0.001; 

// compute entry from previous exit
block.entry_distance = block.exit_distance;
block.entry_position = block.exit_position;

// compute exit from cell ray intersection 
block.exit_distance = intersect_box_exit(block.min_position, block.max_position, ray.origin, ray.inv_direction, block.exit_normal);
block.exit_position = distanceToPosition(block.exit_distance);

// compute span distance
block.span_distance = block.exit_distance - block.entry_distance;

// compute termination condition
block.terminated = block.exit_distance > ray.end_distance;

// compute next coordinates
ivec3 exit_coords = ivec3(round(block.exit_position)) / u_volume.block_size;
ivec3 skipped_coords = block.coords + block.skip_distance * ray.signs;
block.coords = mmix(exit_coords, skipped_coords, block.exit_normal);

// update stats
#if DEBUG_ENABLED == 1

    stats.num_blocks += 1;
    stats.num_fetches += 1;
    stats.num_distance_fetches += 1;

#endif


