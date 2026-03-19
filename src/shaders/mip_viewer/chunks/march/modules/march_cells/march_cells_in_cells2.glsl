
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
mip.value = cubic.values.w;
mip.distance = ray.start_distance;

#if DEBUG_ENABLED == 1

    stats.num_mips += 1;

#endif

// START_MARCH
float exitNudge = ray.spacing * 1e-3;
bool prevNonShadowed = true;

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

    // compute next coordinates
    ivec3 exit_coords = positionToCellCoords(cell.exit_position);
    ivec3 skip_coords = cell.coords + cell.skip_radius * u_ray.signs;
    cell.coords = mmix(exit_coords, skip_coords, cell.exit_normal);

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_cells += 1;

    #endif

    bool consecutiveNonShadowed = prevNonShadowed && !cell.shadowed;
    prevNonShadowed = !cell.shadowed;

    if (!cell.shadowed) 
    {
        // UPDATE_CUBIC     

        vec3 span_position = cell.exit_position - cell.entry_position;

        cubic.values.x = consecutiveNonShadowed ? cubic.values.w : sampleVolume(cell.entry_position);
        cubic.values.y = sampleVolume(cell.entry_position + span_position * (1.0 / 3.0));
        cubic.values.z = sampleVolume(cell.entry_position + span_position * (2.0 / 3.0));
        cubic.values.w = sampleVolume(cell.exit_position);

        cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
        CubicMax cubic_max = cubicMaxFromCoeffs(cubic.coeffs);

        #if DEBUG_ENABLED == 1

            stats.num_volume_fetches += consecutiveNonShadowed ? 3 : 4;
            stats.num_cubics += 1;

        #endif

        // UPDATE_MIP

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
}

mip.position = rayDistanceToPosition(mip.distance); 
mip.gradient = compute_gradient(mip.position, mip.hessian);
mip.curvatures = compute_curvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);


