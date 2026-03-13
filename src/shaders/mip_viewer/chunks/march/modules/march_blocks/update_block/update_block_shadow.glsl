
// compute occupancy
block.shadowed = sample_shadow(block.coords);

// compute min/max positions
block.min_position = vec3((block.coords + 0) * u_volume.block_size) - 0.5;
block.max_position = vec3((block.coords + 1) * u_volume.block_size) - 0.5;  

// compute entry from previous exit
block.entry_distance = block.exit_distance;
block.entry_position = block.exit_position;

// compute exit from cell ray intersection 
block.exit_distance = intersect_box_exit(block.min_position, block.max_position, ray.origin, ray.inv_direction, block.exit_normal);
block.exit_position = ray.origin + ray.direction * block.exit_distance;

// compute span distance
block.span_distance = block.exit_distance - block.entry_distance;

// compute termination condition
block.terminated = block.exit_distance > ray.end_distance;

// compute next coordinates
block.coords += block.exit_normal * ray.signs;

// update stats
#if DEBUG_ENABLED == 1

    stats.num_blocks += 1;
    stats.num_fetches += 1;
    stats.num_distance_fetches += 1;
    
#endif
