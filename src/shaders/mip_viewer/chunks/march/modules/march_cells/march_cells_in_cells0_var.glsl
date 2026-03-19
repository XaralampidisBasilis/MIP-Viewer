
// START_CUBIC_CELL
cell.exit_distance = ray.start_distance;
cell.exit_position = rayDistanceToPosition(cell.exit_distance); 
cell.coords = positionToCellCoords(cell.exit_position);
cubic.values.w = sampleVolume(cell.exit_position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;
    stats.num_cells += 1;

#endif

// START_MIP
mip.distance = ray.start_distance;
mip.value = cubic.values.w;

// update stats
#if DEBUG_ENABLED == 1

    stats.num_mips += 1;

#endif

for (int i = 0; i < MAX_CELLS; i++) 
{
    // UPDATE_CELL

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

    // compute next coordinates
    cell.coords += cell.exit_normal * u_ray.signs;

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_cells += 1;

    #endif

    // UPDATE_CUBIC     
    vec3 span_position = cell.exit_position - cell.entry_position;

    cubic.values.x = cubic.values.w;
    cubic.values.y = sampleVolume(cell.entry_position + span_position * (1.0 / 3.0));
    cubic.values.z = sampleVolume(cell.entry_position + span_position * (2.0 / 3.0));
    cubic.values.w = sampleVolume(cell.exit_position);

    cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;

    #if DEBUG_ENABLED == 1

        stats.num_volume_fetches += 3;

    #endif

    if (any(lessThan(vec4(mip.value), cubic.bernstein_coeffs)))
    {
        // MAXIMIZE CUBIC
        cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
        CubicMax cubic_max = cubicMaxFromCoeffs(cubic.coeffs);

        #if DEBUG_ENABLED == 1

            stats.num_cubics += 1;

        #endif

        // UPDATE_MIP
        mip.distance = cell.entry_distance + cell.span_distance * cubic_max.t;
        mip.value = cubic_max.v;

        #if DEBUG_ENABLED == 1

            stats.num_mips += 1;

        #endif
    }
        
    if (cell.terminated) break;
}

mip.position = rayDistanceToPosition(mip.distance); 
mip.gradient = compute_gradient(mip.position, mip.hessian);
mip.curvatures = compute_curvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);


