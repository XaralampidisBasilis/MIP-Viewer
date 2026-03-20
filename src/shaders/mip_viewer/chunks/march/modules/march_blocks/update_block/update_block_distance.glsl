// compute skip distance
block.step_radius = sample_rgba16ui_distance_fast(block.coords, block.shadowed);
// block.step_radius = sample_rgb32ui_distance_fast(block.coords, block.shadowed);

// compute entry from previous exit
block.entry_distance = block.exit_distance;
block.entry_position = block.exit_position;

// compute exit from cell ray intersection 
block.exit_distance = intersectSkipBlockExit(block.coords, block.step_radius, block.exit_step) + ray.spacing * 0.001;
block.exit_position = rayDistanceToPosition(block.exit_distance);

// compute span distance
block.span_distance = block.exit_distance - block.entry_distance;

// compute termination condition
block.terminated = block.exit_distance > ray.end_distance;

// compute next coordinates
ivec3 exit_coords = positionToBlockCoords(block.exit_position);
ivec3 skipped_coords = block.coords + block.step_radius * u_ray.signs;
block.coords = mmix(exit_coords, skipped_coords, block.exit_step);

// update stats
#if DEBUG_ENABLED == 1

    stats.num_blocks += 1;
    stats.num_distance_fetches += 1;

#endif


