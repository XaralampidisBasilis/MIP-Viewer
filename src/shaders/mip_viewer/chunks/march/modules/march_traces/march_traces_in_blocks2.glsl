
// START_CUBIC_CELL
cell.exit_distance = ray.start_distance;
cell.exit_position = distanceToPosition(cell.exit_distance); 
cell.coords = positionToCellCoords(cell.exit_position);

// START_TRACE
trace.spacing = ray.spacing / 2.0;
trace.distance = trace.spacing * (floor(ray.start_distance / trace.spacing) + ray.phase);
trace.position = distanceToPosition(trace.distance); 
trace.value = sampleVolume(trace.position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;

#endif

// START_MIP
mip.value = trace.value ;
mip.distance = ray.start_distance;

float ray.eps_direction = ray.spacing * 1e-3;

for (int i = 0; i < MAX_CELLS; i++) 
{
    // UPDATE_CELL

    // compute skip distance
    cell.step_radius = sampleDistance5bit(cell.coords, cell.empty);
    // cell.step_radius = sampleDistance8bit(cell.coords, cell.empty);

    // compute entry from previous exit
    cell.entry_distance = cell.exit_distance;
    cell.entry_position = cell.exit_position;

    // compute exit from cell ray intersection 
    cell.exit_distance = intersectSkipCellExit(cell.coords, cell.step_radius, cell.exit_step) + ray.eps_direction;
    cell.exit_position = distanceToPosition(cell.exit_distance);

    // compute span distance
    cell.span_distance = cell.exit_distance - cell.entry_distance;

    // compute termination condition
    cell.terminated = cell.exit_distance > ray.end_distance;

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_cells += 1;

    #endif

    if (!cell.empty) 
    {
        trace.distance = trace.spacing * (floor(cell.entry_distance / trace.spacing) + ray.phase);

        #pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            trace.distance += trace.spacing;
            trace.position = distanceToPosition(trace.distance); 
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

    // compute next coordinates
    ivec3 exit_coords = positionToCellCoords(cell.exit_position);
    ivec3 skip_coords = cell.coords + cell.step_radius * u_ray.signs;
    cell.coords = mmix(exit_coords, skip_coords, cell.exit_step);

}

mip.terminated = mip.distance > ray.end_distance;

if (mip.terminated)
{
    mip.distance = ray.end_distance;
    mip.value = sampleVolume(ray.end_position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;

    #endif
}
 
mip.position = distanceToPosition(mip.distance); 
mip.gradient = computeGradient(mip.position, mip.hessian);
mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);


