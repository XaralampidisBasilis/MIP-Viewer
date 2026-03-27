
// START_BLOCK_IN_RAY

// START_BLOCK
block.coords = positionToBlockCoords(ray.start_position);
block.exit_distance = ray.start_distance;
block.exit_position = ray.start_position; 
block.exit_step = ivec3(0);
block.empty = false;

// START_TRACE_IN_RAY

// START_TRACE_IN_RAY

// set spacing
trace.step_distance = ray.step_distance / float(MAX_TRACES_IN_CELL - 1);

// set position
trace.distance = snapTraceDistanceCeil(ray.start_distance, trace.step_distance, ray.phase);
trace.position = distanceToPosition(trace.distance); 

// set value
trace.value = sampleVolume(trace.position);

// update stats
#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;
    stats.num_traces += 1;

#endif


// START_MIP_IN_TRACE

// START_MIP_IN_TRACE
mip.distance = trace.distance;
mip.value = trace.value;

#if DEBUG_ENABLED == 1

    stats.num_mips += 1;

#endif

// MARCH_BLOCKS
for (int j = 0; j < MAX_BLOCKS; j++) 
{
    // UPDATE_BLOCK_IN_RAY
    
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

    // START_TRACE_IN_BLOCK
    
    // START_TRACE_IN_BLOCK
    trace.distance = snapTraceDistanceFloor(block.entry_distance, trace.step_distance, ray.phase);

    // MARCH_TRACES_IN_BLOCK
    #pragma unroll
    for (int i = 0; i < MAX_TRACES_IN_BLOCK; i++)
    {
        // UPDATE_TRACE_POSITION_IN_BLOCK
        
        // Increment distance
        trace.distance += trace.step_distance;
        
        // Compute position
        trace.position = distanceToPosition(trace.distance); 
        
        // Compute termination condition
        trace.terminated = trace.distance > block.exit_distance || trace.distance > ray.end_distance; 
        
        // update stats
        #if DEBUG_ENABLED == 1
        
            stats.num_traces += 1;
        
        #endif

        // TERMINATE_MARCH_TRACES_IN_BLOCK
        if (trace.terminated) break; 

        // UPDATE_TRACE_VALUE
        
        // Update value
        trace.value = sampleVolume(trace.position);
        
        // update stats
        #if DEBUG_ENABLED == 1
        
            stats.num_volume_fetches += 1;
        
        #endif

        // UPDATE_MIP_IN_TRACE
        
        // Compare trace to mip value
        mip.update = trace.value > mip.value;
        
        if (mip.update)
        {
            // Update value
            mip.distance = trace.distance;
            mip.value = trace.value;
        
            // update stats
            #if DEBUG_ENABLED == 1
        
                stats.num_mips += 1;
        
            #endif
        
        }
        
    }

    // TERMINATE_MARCH_BLOCKS
    if (block.terminated) break;
}

// END_MIP

mip.terminated = mip.distance > ray.end_distance;
mip.position = distanceToPosition(mip.distance); 

mip.gradient = computeGradient(mip.position, mip.hessian);
mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);




