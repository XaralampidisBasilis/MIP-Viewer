
// START_CUBIC_CELL
cell.exit_distance = ray.start_distance;
cell.exit_position = rayDistanceToPosition(cell.exit_distance); 
cell.coords = positionToCellCoords(cell.exit_position);

// START_TRACE
float trace_phase = ray.phase - 0.5;
trace.spacing = ray.spacing / 2.0;
trace.distance = trace.spacing * (ceil(ray.start_distance / trace.spacing) + trace_phase);
trace.position = rayDistanceToPosition(trace.distance); 
trace.value = sampleVolume(trace.position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;

#endif

// START_MIP
mip.value = trace.value ;
mip.distance = ray.start_distance;

bool prevNonShadowed = true;

for (int i = 0; i < MAX_CELLS; i++) 
{
    // UPDATE_CELL

    // compute shadowed
    cell.shadowed = sample_shadow(cell.coords);

    // compute entry from previous exit
    cell.entry_distance = cell.exit_distance;
    cell.entry_position = cell.exit_position;

    // compute exit from cell ray intersection 
    cell.exit_distance = intersectCellExit(cell.coords, cell.exit_step);
    cell.exit_position = rayDistanceToPosition(cell.exit_distance);

    // compute span distance
    cell.span_distance = cell.exit_distance - cell.entry_distance;

    // compute termination condition
    cell.terminated = cell.exit_distance > ray.end_distance;

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_cells += 1;

    #endif

    bool consecutiveNonShadowed = prevNonShadowed && !cell.shadowed;
    prevNonShadowed = !cell.shadowed;

    if (!cell.shadowed) 
    {
        trace.distance = trace.spacing * floor(cell.entry_distance / trace.spacing);

        #pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            trace.distance += trace.spacing;
            trace.position = rayDistanceToPosition(trace.distance); 
            trace.terminated = trace.distance > ray.end_distance; 

            #if DEBUG_ENABLED == 1

                stats.num_traces += 1;

            #endif

            if (trace.distance > cell.exit_distance || trace.terminated) break;

            trace.value = sampleVolume(trace.position);

            if (trace.value > mip.value)
            {
                mip.distance = trace.distance;
                mip.value = trace.value;

                #if DEBUG_ENABLED == 1

                    stats.num_mips += 1;

                #endif
            }

            #if DEBUG_ENABLED == 1

                stats.num_volume_fetches += 1;

            #endif     
        }      
    }

    if (cell.terminated) break;

    cell.coords += cell.exit_step * u_ray.signs;

}

mip.position = rayDistanceToPosition(mip.distance); 
mip.gradient = compute_gradient(mip.position, mip.hessian);
mip.curvatures = compute_curvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);


