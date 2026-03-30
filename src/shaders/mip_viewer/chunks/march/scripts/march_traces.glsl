#if SKIPPING_ENABLED == 1

    #if VARIATION_ENABLED == 1
    
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
    trace.step_distance = u_ray.step_distance / float(MAX_TRACES_IN_CELL - 1);
    
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
    for (int k = 0; k < MAX_BLOCKS; k++) 
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
        
    
        // CONTINUE_OR_BREAK_MARCH_BLOCKS
        if (block.empty)
        {
            if (!block.terminated) continue; else break;    
        }
    
        // START_CELL_IN_BLOCK
        
        #if BLOCK_SIZE != 1
        
        // START_CELL_IN_BLOCK
        cell.coords = startCellCoordsInBlock(block.coords, block.entry_step, block.entry_position + ray.eps_direction);
        cell.far_distances = cellCoordsToFarDistances(cell.coords);
        
        cell.exit_distance = block.entry_distance;
        cell.exit_position = block.entry_position; 
        cell.exit_step = ivec3(0);
        
        #endif
    
        // START_TRACE_IN_BLOCK
        
        // START_TRACE_IN_CELL
        if(block.prev_empty)
        {
            // Increment distance
            trace.distance = block.entry_distance;
        
            // Compute position
            trace.position = block.entry_position; 
        
            // Update value
            trace.value = sampleVolume(trace.position);
        
            #if DEBUG_ENABLED == 1
        
                stats.num_volume_fetches += 1;
                stats.num_traces += 1;
            
            #endif
        }
          
        // MARCH_CELLS_IN_BLOCK
        #pragma unroll
        for (int j = 0; j < MAX_CELLS_IN_BLOCK; j++)
        {
            // UPDATE_CELL_IN_BLOCK
            // UPDATE_CELL_IN_BLOCK
            
            #if BLOCK_SIZE != 1
            
            // compute far distances
            cell.coords = advanceCellCoords(cell.coords, cell.exit_step);
            cell.far_distances = advanceCellFarDistances(cell.far_distances, cell.exit_step);
            
            // compute entry from previous exit
            cell.entry_distance = cell.exit_distance;
            cell.entry_position = cell.exit_position;
            
            // compute exit from cell ray intersection 
            cell.exit_distance = intersectCellFarDistances(cell.far_distances, cell.exit_step);
            cell.exit_position = distanceToPosition(cell.exit_distance);
            
            // compute span distance
            cell.span_distance = cell.exit_distance - cell.entry_distance;
            
            // compute termination condition
            cell.terminated = cell.exit_distance > block.exit_distance - ray.eps_distance;
            if (cell.terminated)
            {
                cell.exit_distance = block.exit_distance;
                cell.exit_position = block.exit_position;
            }
                
            // update stats
            #if DEBUG_ENABLED == 1
            
                stats.num_cells += 1;
            
            #endif
            
            #else
            
            // copy cell from block
            cell.entry_distance = block.entry_distance;
            cell.entry_position = block.entry_position;
            
            cell.exit_distance = block.exit_distance;
            cell.exit_position = block.exit_position;
            
            cell.terminated = true;
            
            #endif
            
    
            // MARCH_TRACES_IN_CELL
            #pragma unroll
            for (int i = 1; i < MAX_TRACES_IN_CELL; i++)
            {
                // UPDATE_TRACE_IN_CELL
                
                // point inside the cell entry and exit
                float t = float(i) / float(MAX_TRACES_IN_CELL - 1);
                
                // Increment distance
                trace.distance = mix(cell.entry_distance, cell.exit_distance, t);
                
                // Compute position
                trace.position = distanceToPosition(trace.distance); 
                
                // Update value
                trace.value = sampleVolume(trace.position);
                
                // Compute termination condition
                trace.terminated = trace.distance > ray.end_distance; 
                
                // update stats
                #if DEBUG_ENABLED == 1
                
                    stats.num_volume_fetches += 1;
                    stats.num_traces += 1;
                
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
    
            // BREAK_MARCH_CELLS_IN_BLOCK
            if (cell.terminated) break; 
        }
    
        // BREAK_MARCH_BLOCKS
        if (block.terminated) break;
    }
    
    // END_MIP
    
    mip.terminated = mip.distance > ray.end_distance;
    mip.position = distanceToPosition(mip.distance); 
    
    mip.gradient = computeGradient(mip.position, mip.hessian);
    mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
    mip.normal = normalize(mip.gradient);
    
    
    
    
    #else
    
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
    trace.step_distance = u_ray.step_distance / float(MAX_TRACES_IN_CELL - 1);
    
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
    
        // START_TRACE_IN_BLOCK
        
        // START_TRACE_IN_BLOCK
        trace.distance = snapTraceDistanceFloor(block.entry_distance, trace.step_distance, ray.phase);
        
        // Compute position
        trace.position = distanceToPosition(trace.distance); 
        
        // update stats
        #if DEBUG_ENABLED == 1
        
            stats.num_traces += 1;
        
        #endif
    
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
            trace.terminated = trace.distance > block.exit_distance; 
            
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
    
    
    
    
    #endif

#else

    #if VARIATION_ENABLED == 1
    
    // START_BLOCK_IN_RAY
    
    // START_CELL_IN_RAY
    cell.coords = positionToCellCoords(ray.start_position);
    cell.far_distances = cellCoordsToFarDistances(cell.coords);
    
    cell.exit_distance = ray.start_distance;
    cell.exit_position = ray.start_position; 
    cell.exit_step = ivec3(0);
    
    // START_TRACE_IN_RAY  
    
    // START_TRACE_IN_RAY
    
    // set spacing
    trace.step_distance = u_ray.step_distance / float(MAX_TRACES_IN_CELL - 1);
    
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
    
    // MARCH_CELLS
    for (int j = 0; j < MAX_CELLS; j++) 
    {
        // UPDATE_CELL_IN_RAY
        // UPDATE_CELL_IN_RAY
        
        // compute far distances
        cell.coords = advanceCellCoords(cell.coords, cell.exit_step);
        cell.far_distances = advanceCellFarDistances(cell.far_distances, cell.exit_step);
        
        // compute entry from previous exit
        cell.entry_distance = cell.exit_distance;
        cell.entry_position = cell.exit_position;
        
        // compute exit from cell ray intersection 
        cell.exit_distance = intersectCellFarDistances(cell.far_distances, cell.exit_step);
        cell.exit_position = distanceToPosition(cell.exit_distance);
        
        // compute span distance
        cell.span_distance = cell.exit_distance - cell.entry_distance;
        
        // compute termination condition
        cell.terminated = cell.exit_distance > ray.end_distance - ray.eps_distance;
        if (cell.terminated)
        {
            cell.exit_distance = ray.end_distance;
            cell.exit_position = ray.end_position;
        }
        
        // update stats
        #if DEBUG_ENABLED == 1
        
            stats.num_cells += 1;
        
        #endif
        
    
        // MARCH_TRACES_IN_CELL
        #pragma unroll
        for (int i = 1; i < MAX_TRACES_IN_CELL; i++)
        {
            // UPDATE_TRACE_IN_CELL
            
            // point inside the cell entry and exit
            float t = float(i) / float(MAX_TRACES_IN_CELL - 1);
            
            // Increment distance
            trace.distance = mix(cell.entry_distance, cell.exit_distance, t);
            
            // Compute position
            trace.position = distanceToPosition(trace.distance); 
            
            // Update value
            trace.value = sampleVolume(trace.position);
            
            // Compute termination condition
            trace.terminated = trace.distance > ray.end_distance; 
            
            // update stats
            #if DEBUG_ENABLED == 1
            
                stats.num_volume_fetches += 1;
                stats.num_traces += 1;
            
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
    
        // BREAK_MARCH_CELLS
        if (cell.terminated) break; 
    }
    
    // END_MIP
    
    mip.terminated = mip.distance > ray.end_distance;
    mip.position = distanceToPosition(mip.distance); 
    
    mip.gradient = computeGradient(mip.position, mip.hessian);
    mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
    mip.normal = normalize(mip.gradient);
    
    
    
    
    #else
    
    // START_TRACE_IN_RAY
    
    // START_TRACE_IN_RAY
    
    // set spacing
    trace.step_distance = u_ray.step_distance / float(MAX_TRACES_IN_CELL - 1);
    
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
    
    // MARCH_TRACES
    for (int i = 0; i < MAX_TRACES; i++) 
    {
        // UPDATE_TRACE_POSITION_IN_RAY
        
        // Increment distance
        trace.distance += trace.step_distance;
        
        // Compute position
        trace.position = distanceToPosition(trace.distance); 
        
        // Compute termination condition
        trace.terminated = trace.distance > ray.end_distance; 
        
        // update stats
        #if DEBUG_ENABLED == 1
        
            stats.num_traces += 1;
        
        #endif
    
        // TERMINATE_MARCH_TRACES
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
    
    // END_MIP
    
    mip.terminated = mip.distance > ray.end_distance;
    mip.position = distanceToPosition(mip.distance); 
    
    mip.gradient = computeGradient(mip.position, mip.hessian);
    mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
    mip.normal = normalize(mip.gradient);
    
    
    
    
    #endif

#endif

