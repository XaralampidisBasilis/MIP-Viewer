
// UPDATE_BLOCK

// Choose next block coords from either geometric exit or skip step
block.coords = advanceBlockCoords(block.coords, block.exit_step, block.step_radius, block.exit_position + ray.eps_direction);

// Read skip radius and shadow flag for the current block
block.prev_empty = block.empty;
block.step_radius = sampleDistance(block.coords, block.empty);

// Current entry is the previous step's exit
block.entry_distance = block.exit_distance;
block.entry_position = block.exit_position;
block.entry_step = block.exit_step;

// Find exit point of the current skip block
block.exit_distance = intersectBlockExit(block.coords, block.step_radius, block.exit_step);
block.exit_position = distanceToPosition(block.exit_distance);

// Distance covered inside this block span
block.span_distance = block.exit_distance - block.entry_distance;

// Stop once the ray exit goes beyond the ray end
block.terminated = block.exit_distance > ray.end_distance - ray.eps_distance;
if (block.terminated)
{
    block.exit_distance = ray.end_distance;
    block.exit_position = ray.end_position;
}

// update stats
#if DEBUG_ENABLED == 1

    stats.num_distance_fetches += 1;
    stats.num_blocks += 1;

#endif