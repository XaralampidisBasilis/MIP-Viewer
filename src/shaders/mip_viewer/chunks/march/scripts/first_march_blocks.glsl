
// start block at ray start

// START_BLOCK
block.coords = positionToBlockCoords(ray.start_position);

block.exit_distance = ray.start_distance;
block.exit_position = ray.start_position; 
block.exit_step = ivec3(0);

block.empty = false;
block.prev_empty = false;

// start mip at the ray start

// UPDATE_MIP_IN_RAY
float ray_value = sampleVolume(ray.start_position);
mip.update = shouldUpdateMip(mip.value, ray_value);

if (mip.update) 
{
    mip.distance = ray.start_distance;
    mip.value = ray_value;

    #if DEBUG_ENABLED == 1

        stats.num_mips += 1;

    #endif
}

// START_MARCH
for (int i = 0; i < MAX_BLOCKS; i++) 
{
    // Update block and get if its empty and what is the step distance we can take
    
    #if SKIPPING_METHOD == 0
    
    // UPDATE_BLOCK
    
    // compute next coordinates
    block.coords = advanceBlockCoords(block.coords, block.exit_step);
    
    // compute empty
    block.prev_empty = block.empty;
    block.empty = sampleShadow(block.coords);
    
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
    
    #elif SKIPPING_METHOD == 1
    
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
    #endif    
    

    // CONTINUE_OR_TERMINATE_MARCH_BLOCKS
    if (block.empty)
    {
        if (!block.terminated) continue; else break;    
    }

    // Update mip with the block entry point
    
    // UPDATE_MIP_IN_BLOCK
    float block_value = sampleVolume(block.entry_position);
    mip.update = shouldUpdateMip(mip.value, block_value);
    
    if (mip.update) 
    {
        mip.distance = block.entry_distance;
        mip.value = block_value;
    
        #if DEBUG_ENABLED == 1
    
            stats.num_mips += 1;
    
        #endif
    }

    if (block.terminated) break;
}

// END_RAY_IN_MIP

// END_RAY_IN_MIP
ray.end_distance = mip.distance + ray.eps_distance;
ray.end_position = distanceToPosition(ray.end_distance);

ray.span_distance = ray.end_distance - ray.start_distance;



