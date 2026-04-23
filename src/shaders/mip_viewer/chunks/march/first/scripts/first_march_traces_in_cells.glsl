
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
mip.update = shouldUpdateMip(mip.value, trace.value);

if (mip.update)
{
    mip.distance = trace.distance;
    mip.value = trace.value;

    #if DEBUG_ENABLED == 1

        stats.num_mips += 1;

    #endif

}

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
        
        // UPDATE_MIP_IN_TRACE
        mip.update = shouldUpdateMip(mip.value, trace.value);
        
        if (mip.update)
        {
            mip.distance = trace.distance;
            mip.value = trace.value;
        
            #if DEBUG_ENABLED == 1
        
                stats.num_mips += 1;
        
            #endif
        
        }
        
    }

    // BREAK_MARCH_CELLS
    if (cell.terminated) break; 
}

// END_RAY_IN_MIP

// END_RAY_IN_MIP
ray.end_distance = mip.distance + ray.eps_distance;
ray.end_position = distanceToPosition(ray.end_distance);

ray.span_distance = ray.end_distance - ray.start_distance;



