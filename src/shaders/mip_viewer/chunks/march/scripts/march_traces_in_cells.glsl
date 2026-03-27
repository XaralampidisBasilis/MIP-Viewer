
// START_CELL_IN_RAY
cell.coords = positionToCellCoords(ray.start_position);
cell.far_distances = cellCoordsToFarDistances(cell.coords);

cell.exit_distance = ray.start_distance;
cell.exit_position = ray.start_position; 
cell.exit_step = ivec3(0);

// START_TRACE_IN_RAY  
trace.step_distance = ray.step_distance / float(MAX_TRACES_IN_CELL - 1);

trace.distance = snapTraceDistanceCeil(ray.start_distance, trace.step_distance, ray.phase);
trace.position = distanceToPosition(trace.distance); 

trace.value = sampleVolume(trace.position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;
    stats.num_traces += 1;

#endif


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
    cell.coords = advanceCellCoords(cell.coords, cell.exit_step);
    cell.far_distances = advanceCellFarDistances(cell.far_distances, cell.exit_step);
    
    cell.entry_distance = cell.exit_distance;
    cell.entry_position = cell.exit_position;
    
    cell.exit_distance = intersectCellFarDistances(cell.far_distances, cell.exit_step);
    cell.exit_position = distanceToPosition(cell.exit_distance);
    
    cell.span_distance = cell.exit_distance - cell.entry_distance;
    
    cell.terminated = cell.exit_distance > ray.end_distance - ray.eps_distance;

    #if DEBUG_ENABLED == 1
    
        stats.num_cells += 1;
    
    #endif
    
    // MARCH_TRACES_IN_CELL
    #pragma unroll
    for (int i = 1; i < MAX_TRACES_IN_CELL; i++)
    {
        // UPDATE_TRACE_IN_CELL
        
        float t = float(i) / float(MAX_TRACES_IN_CELL - 1);
        
        trace.distance = mix(cell.entry_distance, cell.exit_distance, t);
        trace.position = distanceToPosition(trace.distance); 
        
        trace.value = sampleVolume(trace.position);
        trace.terminated = trace.distance > ray.end_distance; 
        
        // update stats
        #if DEBUG_ENABLED == 1
        
            stats.num_volume_fetches += 1;
            stats.num_traces += 1;
        
        #endif

        // UPDATE_MIP_IN_TRACE
        mip.update = trace.value > mip.value;
        
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

// END_MIP
mip.terminated = mip.distance > ray.end_distance;
mip.position = distanceToPosition(mip.distance); 

mip.gradient = computeGradient(mip.position, mip.hessian);
mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);




