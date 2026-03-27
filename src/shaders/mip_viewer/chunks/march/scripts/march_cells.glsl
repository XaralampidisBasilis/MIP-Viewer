#if SKIPPING_ENABLED == 1

    
    // start block at ray start
    
    // START_BLOCK
    block.coords = positionToBlockCoords(ray.start_position);
    block.exit_distance = ray.start_distance;
    block.exit_position = ray.start_position; 
    block.exit_step = ivec3(0);
    block.empty = false;
    
    // start cubic at the ray start
    
    // START_CUBIC_IN_RAY
    cubic.values.w = sampleVolume(ray.start_position);
    
    #if DEBUG_ENABLED == 1
    
        stats.num_volume_fetches += 1;
    
    #endif
    
    // start mip at the ray start
    
    // START_MIP_IN_CUBIC
    mip.distance = ray.start_distance;
    mip.value = cubic.values.w;
    
    #if DEBUG_ENABLED == 1
    
        stats.num_mips += 1;
    
    #endif
    
    // START_MARCH
    for (int j = 0; j < MAX_BLOCKS; j++) 
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
    
        // Start cell at the block entry
        
        #if BLOCK_SIZE != 1
        
        // START_CELL_IN_BLOCK
        cell.coords = startCellCoordsInBlock(block.coords, block.entry_step, block.entry_position + ray.eps_direction);
        cell.far_distances = cellCoordsToFarDistances(cell.coords);
        
        cell.exit_distance = block.entry_distance;
        cell.exit_position = block.entry_position; 
        cell.exit_step = ivec3(0);
        
        #endif
    
        // Start cubic at the block entry and reuse sample if previous was not empty
        
        
        // START_CUBIC_IN_BLOCK
        if(block.prev_empty)
        {
            cubic.values.w = sampleVolume(block.entry_position);
        
            #if DEBUG_ENABLED == 1
        
                stats.num_volume_fetches += 1;
            
            #endif
        }
          
        // Start cell march inside the current non empty block until we escape
        #pragma unroll
        for (int i = 0; i < MAX_CELLS_IN_BLOCK; i++)
        {
            // update cell based on the previous one
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
            cell.terminated = 
                cell.exit_distance > block.exit_distance - ray.eps_distance || 
                cell.exit_distance > ray.end_distance - ray.eps_distance;
            
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
            
    
            // Reconstruct the cubic polynomial inside the cell entry and exit
            
            // UPDATE_CUBIC_IN_CELL
            vec3 span_vector = cell.exit_position - cell.entry_position;
            
            cubic.values.x = cubic.values.w;
            cubic.values.y = sampleVolume(cell.entry_position + span_vector * (1.0 / 3.0));
            cubic.values.z = sampleVolume(cell.entry_position + span_vector * (2.0 / 3.0));
            cubic.values.w = sampleVolume(cell.exit_position);
            
            #if DEBUG_ENABLED == 1
            
                stats.num_volume_fetches += 3;
            
            #endif
    
            // Maximize the cubic inside the cell 
            
            #if BERNSTEIN_ENABLED == 1
            
            // Cull with Bernstein coefficients before the full cubic maximize step.
            cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;
            cubic.maximize = any(greaterThan(cubic.bernstein_coeffs, vec4(mip.value)));
            
            if (cubic.maximize)
            {
                // MAXIMIZE_CUBIC
                cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
                CubicMax cubic_max = cubicMaxOnUnitInterval(cubic.coeffs, cubic.values.x, cubic.values.w);
            
                cubic.max_value = cubic_max.value;
                cubic.argmax_point = cubic_max.point;
            
                #if DEBUG_ENABLED == 1
            
                    stats.num_maxima += 1;
            
                #endif
            }
            #else
            
            // MAXIMIZE_CUBIC
            cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
            CubicMax cubic_max = cubicMaxOnUnitInterval(cubic.coeffs, cubic.values.x, cubic.values.w);
            
            cubic.max_value = cubic_max.value;
            cubic.argmax_point = cubic_max.point;
            
            #if DEBUG_ENABLED == 1
            
                stats.num_maxima += 1;
            
            #endif
            #endif
            
    
            // Update mip based on the max cubic value
            
            
            // UPDATE_MIP_IN_CUBIC
            mip.update = cubic.max_value > mip.value;
            
            if (mip.update) 
            {
                mip.distance = mix(cell.entry_distance, cell.exit_distance, cubic.argmax_point);
                mip.value = cubic.max_value;
            
                #if DEBUG_ENABLED == 1
            
                    stats.num_mips += 1;
            
                #endif
            }
    
            if (cell.terminated) break; 
        }
    
        if (block.terminated) break;
    }
    
    // END_MIP
    
    mip.terminated = mip.distance > ray.end_distance;
    mip.position = distanceToPosition(mip.distance); 
    
    mip.gradient = computeGradient(mip.position, mip.hessian);
    mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
    mip.normal = normalize(mip.gradient);
    
    
    
    

#else

    
    // start block at ray start
    
    // START_CELL_IN_RAY
    cell.coords = positionToCellCoords(ray.start_position);
    cell.far_distances = cellCoordsToFarDistances(cell.coords);
    
    cell.exit_distance = ray.start_distance;
    cell.exit_position = ray.start_position; 
    cell.exit_step = ivec3(0);
    
    // start cubic at the ray start
    
    // START_CUBIC_IN_RAY
    cubic.values.w = sampleVolume(ray.start_position);
    
    #if DEBUG_ENABLED == 1
    
        stats.num_volume_fetches += 1;
    
    #endif
    
    // start mip at the ray start
    
    // START_MIP_IN_CUBIC
    mip.distance = ray.start_distance;
    mip.value = cubic.values.w;
    
    #if DEBUG_ENABLED == 1
    
        stats.num_mips += 1;
    
    #endif
    
    // START_MARCH
    for (int i = 0; i < MAX_CELLS; i++) 
    {
        // update cell based on the previous one
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
        
        // update stats
        #if DEBUG_ENABLED == 1
        
            stats.num_cells += 1;
        
        #endif
        
    
        // Reconstruct the cubic polynomial inside the cell entry and exit
        
        // UPDATE_CUBIC_IN_CELL
        vec3 span_vector = cell.exit_position - cell.entry_position;
        
        cubic.values.x = cubic.values.w;
        cubic.values.y = sampleVolume(cell.entry_position + span_vector * (1.0 / 3.0));
        cubic.values.z = sampleVolume(cell.entry_position + span_vector * (2.0 / 3.0));
        cubic.values.w = sampleVolume(cell.exit_position);
        
        #if DEBUG_ENABLED == 1
        
            stats.num_volume_fetches += 3;
        
        #endif
    
        // Maximize the cubic inside the cell 
        
        #if BERNSTEIN_ENABLED == 1
        
        // Cull with Bernstein coefficients before the full cubic maximize step.
        cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;
        cubic.maximize = any(greaterThan(cubic.bernstein_coeffs, vec4(mip.value)));
        
        if (cubic.maximize)
        {
            // MAXIMIZE_CUBIC
            cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
            CubicMax cubic_max = cubicMaxOnUnitInterval(cubic.coeffs, cubic.values.x, cubic.values.w);
        
            cubic.max_value = cubic_max.value;
            cubic.argmax_point = cubic_max.point;
        
            #if DEBUG_ENABLED == 1
        
                stats.num_maxima += 1;
        
            #endif
        }
        #else
        
        // MAXIMIZE_CUBIC
        cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
        CubicMax cubic_max = cubicMaxOnUnitInterval(cubic.coeffs, cubic.values.x, cubic.values.w);
        
        cubic.max_value = cubic_max.value;
        cubic.argmax_point = cubic_max.point;
        
        #if DEBUG_ENABLED == 1
        
            stats.num_maxima += 1;
        
        #endif
        #endif
        
    
        // Update mip based on the max cubic value
        
        
        // UPDATE_MIP_IN_CUBIC
        mip.update = cubic.max_value > mip.value;
        
        if (mip.update) 
        {
            mip.distance = mix(cell.entry_distance, cell.exit_distance, cubic.argmax_point);
            mip.value = cubic.max_value;
        
            #if DEBUG_ENABLED == 1
        
                stats.num_mips += 1;
        
            #endif
        }
    
        if (cell.terminated) break; 
    }
    
    // END_MIP
    
    mip.terminated = mip.distance > ray.end_distance;
    mip.position = distanceToPosition(mip.distance); 
    
    mip.gradient = computeGradient(mip.position, mip.hessian);
    mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
    mip.normal = normalize(mip.gradient);
    
    
    
    
    
#endif

