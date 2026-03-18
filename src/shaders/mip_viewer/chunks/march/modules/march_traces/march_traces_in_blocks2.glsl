
// START_CUBIC_CELL
cell.exit_distance = ray.start_distance;
cell.exit_position = rayDistanceToPosition(cell.exit_distance); 
cell.coords = positionToCellCoords(cell.exit_position);

// START_TRACE
trace.spacing = ray.spacing / 2.0;
trace.distance = trace.spacing * (floor(ray.start_distance / trace.spacing) + ray.phase);
trace.position = rayDistanceToPosition(trace.distance); 
trace.value = sampleVolume(trace.position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;
    stats.num_fetches += 1;

#endif

// START_MIP
mip.value = trace.value ;
mip.distance = ray.start_distance;

float exitNudge = ray.spacing * 1e-3;

for (int i = 0; i < MAX_CELLS; i++) 
{
    // UPDATE_CELL

    // compute skip distance
    cell.skip_radius = sample_rgba16ui_distance_fast(cell.coords, cell.shadowed);
    // cell.skip_radius = sample_rgb32ui_distance_fast(cell.coords, cell.shadowed);

    // compute entry from previous exit
    cell.entry_distance = cell.exit_distance;
    cell.entry_position = cell.exit_position;

    // compute exit from cell ray intersection 
    cell.exit_distance = intersectSkipCellExit(cell.coords, cell.skip_radius, cell.exit_normal) + exitNudge;
    cell.exit_position = rayDistanceToPosition(cell.exit_distance);

    // compute span distance
    cell.span_distance = cell.exit_distance - cell.entry_distance;

    // compute termination condition
    cell.terminated = cell.exit_distance > ray.end_distance;

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_fetches += 1;
        stats.num_cells += 1;

    #endif

    if (!cell.shadowed) 
    {
        trace.distance = trace.spacing * (floor(cell.entry_distance / trace.spacing) + ray.phase);

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
                stats.num_fetches += 1;

            #endif     
        }   
    }

    if (cell.terminated) break;

    // compute next coordinates
    ivec3 exit_coords = positionToCellCoords(cell.exit_position);
    ivec3 skip_coords = cell.coords + cell.skip_radius * u_ray.signs;
    cell.coords = mmix(exit_coords, skip_coords, cell.exit_normal);

}

mip.position = rayDistanceToPosition(mip.distance); 
mip.gradient = compute_gradient(mip.position, mip.hessian);
mip.curvatures = compute_curvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);


