
// UPDATE_BLOCK

// Choose next block coords from either geometric exit or skip step
block.coords = advanceBlockCoords(block.coords, -block.entry_step, block.step_radius, block.entry_position - ray.eps_direction);

// Read skip radius and shadow flag for the current block
block.step_radius = sampleDistance(block.coords, block.empty);

// Current entry is the previous step's entry
block.exit_distance = block.entry_distance;
block.exit_position = block.entry_position;
block.exit_step = block.entry_step;

// Find exit point of the current skip block
block.entry_distance = intersectBlockEntry(block.coords, block.step_radius, block.entry_step);
block.entry_position = distanceToPosition(block.entry_distance);

// Distance covered inside this block span
block.span_distance = block.exit_distance - block.entry_distance;

// Stop once the ray exit goes beyond the ray end
block.terminated = block.entry_distance < ray.start_distance + ray.eps_distance;
if (block.terminated)
{
    block.entry_distance = ray.start_distance;
    block.entry_position = ray.start_position;
}

// update stats
#if DEBUG_ENABLED == 1

    stats.num_distance_fetches += 1;
    stats.num_blocks += 1;

#endif