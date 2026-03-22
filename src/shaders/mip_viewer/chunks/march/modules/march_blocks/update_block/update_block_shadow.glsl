
// compute occupancy
block.step_radius = 1;
block.shadowed = sampleShadow(block.coords);

// compute entry from previous exit
block.entry_distance = block.exit_distance;
block.entry_position = block.exit_position;

// compute exit from cell ray intersection 
block.exit_distance = intersectSkipBlockExit(block.coords, block.step_radius, block.exit_step);
block.exit_position = rayDistanceToPosition(block.exit_distance);

// compute span distance
block.span_distance = block.exit_distance - block.entry_distance;

// compute termination condition
block.terminated = block.exit_distance > ray.end_distance;

// compute next coordinates
block.coords += block.exit_step * u_ray.signs;

// update stats
#if DEBUG_ENABLED == 1

    stats.num_blocks += 1;
    stats.num_distance_fetches += 1;
    
#endif
