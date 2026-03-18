
// START_CUBIC_CELL
cell.exit_distance = ray.start_distance;
cell.exit_position = rayDistanceToPosition(cell.exit_distance); 
cell.coords = positionToCellCoords(cell.exit_position);
cubic.values.w = sampleVolume(cell.exit_position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;
    stats.num_fetches += 1;

#endif

// START_MIP
mip.value = cubic.values.w;
mip.distance = ray.start_distance;

bool prevNonShadowed = true;

for (int i = 0; i < u_debug.max_cells; i++) 
{
    // UPDATE_CELL

    // compute shadowed
    cell.shadowed = sample_shadow(cell.coords);

    // compute entry from previous exit
    cell.entry_distance = cell.exit_distance;
    cell.entry_position = cell.exit_position;

    // compute exit from cell ray intersection 
    cell.exit_distance = intersectCellExit(cell.coords, cell.exit_normal);
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

    bool consecutiveNonShadowed = prevNonShadowed && !cell.shadowed;
    prevNonShadowed = !cell.shadowed;

    if (!cell.shadowed) 
    {
        // UPDATE_CUBIC     

        cubic.values.x = consecutiveNonShadowed ? cubic.values.w : sampleVolume(cell.entry_position);

        #if DEBUG_ENABLED == 1

            stats.num_volume_fetches += consecutiveNonShadowed ? 0 : 1;
            stats.num_fetches += consecutiveNonShadowed ? 0 : 1;

        #endif

        vec3 span_position = cell.exit_position - cell.entry_position;
        
        cubic.values.y = sampleVolume(cell.entry_position + span_position * (1.0 / 3.0));
        cubic.values.z = sampleVolume(cell.entry_position + span_position * (2.0 / 3.0));
        cubic.values.w = sampleVolume(cell.exit_position);

        #if DEBUG_ENABLED == 1

            stats.num_volume_fetches += 3;
            stats.num_fetches += 3;

        #endif

        // UPDATE_MIP

        cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
        CubicMax cubic_max = cubicMaxFromCoeffs(cubic.coeffs);

        if (mip.value < cubic_max.v) 
        {
            // mip.distance = mix(cell.entry_distance, cell.exit_distance, cubic_max.t);
            mip.distance = cell.entry_distance + cell.span_distance * cubic_max.t;
            mip.value = cubic_max.v;

            #if DEBUG_ENABLED == 1

                stats.num_mips += 1;

            #endif
        }

    }

    if (cell.terminated) break;

    // compute next coordinates
    cell.coords += cell.exit_normal * u_ray.signs;

}

mip.position = rayDistanceToPosition(mip.distance); 
mip.gradient = compute_gradient(mip.position, mip.hessian);
mip.curvatures = compute_curvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);


