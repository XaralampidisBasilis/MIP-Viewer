
// UPDATE_BLOCK

// compute next coordinates
block.coords = advanceBlockCoords(block.coords, block.exit_step);

// compute empty
block.prev_empty = block.empty;
// block.empty = sampleShadow1bit(block.coords);
// block.empty = sampleShadow5bit(block.coords);
block.empty = sampleShadow8bit(block.coords);

// compute entry from previous exit
block.entry_distance = block.exit_distance;
block.entry_position = block.exit_position;
block.entry_step = block.exit_step;

// compute exit from block ray intersection 
block.exit_distance = intersectBlockExit(block.coords, block.exit_step);
block.exit_position = distanceToPosition(block.exit_distance);

// compute span distance
block.span_distance = block.exit_distance - block.entry_distance;

// compute termination condition
block.terminated = block.exit_distance > ray.end_distance;

// update stats
#if DEBUG_ENABLED == 1

    stats.num_distance_fetches += 1;
    stats.num_blocks += 1;

#endif