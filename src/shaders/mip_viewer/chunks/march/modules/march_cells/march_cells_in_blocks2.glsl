
// START_CUBIC_CELL
cell.exit_distance = ray.start_distance;
cell.exit_position = rayDistanceToPosition(cell.exit_distance); 
cell.coords = positionToCellCoords(cell.exit_position);

// START_MIP
mip.value = 0.0;
mip.distance = ray.start_distance;

for (int i = 0; i < MAX_CELLS; i++) 
{
    // UPDATE_CELL

    // compute skip distance
    cell.skip_distance = sample_rgba16ui_distance_fast(cell.coords, cell.shadowed);
    // cell.skip_distance = sample_rgb32ui_distance_fast(cell.coords, cell.shadowed);

    // compute entry from previous exit
    cell.entry_distance = cell.exit_distance;
    cell.entry_position = cell.exit_position;

    // compute exit from cell ray intersection 
    cell.exit_distance = intersectSkipCellExit(cell.coords, cell.skip_distance, cell.exit_normal) + 0.001;
    cell.exit_position = rayDistanceToPosition(cell.exit_distance);

    // compute span distance
    cell.span_distance = cell.exit_distance - cell.entry_distance;

    // compute termination condition
    cell.terminated = cell.exit_distance > ray.end_distance;

    // compute next coordinates
    ivec3 exit_coords = positionToCellCoords(cell.exit_position);
    ivec3 skip_coords = cell.coords + cell.skip_distance * u_ray.signs;
    cell.coords = mmix(exit_coords, skip_coords, cell.exit_normal);

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_fetches += 1;
        stats.num_cells += 1;

    #endif

    if (!cell.shadowed) 
    {
        // UPDATE_CUBIC     
        vec3 dp = cell.exit_position - cell.entry_position;

        cubic.values[0] = sampleVolume(cell.entry_position);
        cubic.values[1] = sampleVolume(cell.entry_position + dp * (1.0 / 3.0));
        cubic.values[2] = sampleVolume(cell.entry_position + dp * (2.0 / 3.0));
        cubic.values[3] = sampleVolume(cell.exit_position);

        // update stats
        #if DEBUG_ENABLED == 1

            stats.num_volume_fetches += 4;
            stats.num_fetches += 4;

        #endif

        // UPDATE_MIP

        // compute cubic coefficients
        cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
        CubicMax cubic_max = cubicMaxFromCoeffs(cubic.coeffs);

        if (mip.value < cubic_max.v) 
        {
            mip.distance = mix(cell.entry_distance, cell.exit_distance, cubic_max.t);
            mip.value = cubic_max.v;

            #if DEBUG_ENABLED == 1

                stats.num_mips += 1;

            #endif
        }

        // UPDATE_MIP

        // // compute cubic bernstein coefficients
        // vec4 coeffs = cubic.values * CUBIC_INV_BERNSTEIN;
        // cubic.bernstein_coeffs = coeffs;

        // if (mip.value < coeffs.x || mip.value < coeffs.y || mip.value < coeffs.z || mip.value < coeffs.w) 
        // {
        //     // compute cubic coefficients
        //     cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
        //     CubicMax c_max = cubicMaxFromCoeffs(cubic.coeffs);

        //     mip.distance = mix(cell.entry_distance, cell.exit_distance, c_max.t);
        //     mip.value = c_max.v;

        //     #if DEBUG_ENABLED == 1

        //         stats.num_mips += 1;

        //     #endif
        // }
    }

    if (cell.terminated) break;
}

#include "./end_mip"

