
// START_BLOCK
block.exit_distance = ray.start_distance;
block.exit_position = ray.start_position; 
block.coords = positionToBlockCoords(ray.start_position);

// START_CUBIC
cubic.values.w = sampleVolume(ray.start_position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;

#endif

// START_MIP
mip.distance = ray.start_distance;
mip.value = cubic.values.w;

#if DEBUG_ENABLED == 1

    stats.num_mips += 1;

#endif

// START_MARCH
float exitNudge = ray.spacing * 1e-3;
bool prevNonShadowed = true;

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    // UPDATE_BLOCK

    // compute shadowed
    block.shadowed = sample_shadow(block.coords);

    // compute entry from previous exit
    block.entry_distance = block.exit_distance;
    block.entry_position = block.exit_position;

    // compute exit from block ray intersection 
    block.exit_distance = intersectBlockExit(block.coords, block.exit_step);
    block.exit_position = rayDistanceToPosition(block.exit_distance);

    // compute span distance
    block.span_distance = block.exit_distance - block.entry_distance;

    // compute termination condition
    block.terminated = block.exit_distance > ray.end_distance;

    // compute next coordinates
    block.coords = advanceBlockCoords(block.coords, block.exit_step);

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_blocks += 1;

    #endif

    bool consecutiveNonShadowed = prevNonShadowed && !block.shadowed;
    prevNonShadowed = !block.shadowed;

    if (block.shadowed && !block.terminated) continue;

    // START_CELL_TO_BLOCK
    cell.exit_distance = block.entry_distance;
    cell.exit_position = block.entry_position; 
    cell.coords = positionToCellCoords(block.entry_position);

    // START_CUBIC
    if (!consecutiveNonShadowed)
    {
        cubic.values.w = sampleVolume(block.entry_position);

        #if DEBUG_ENABLED == 1

            stats.num_volume_fetches += 1;
        
        #endif
    }
      
    // START_MARCH_IN_BLOCK
    for (int i = 0; i < MAX_CELLS_IN_BLOCK; i++)
    {
        // UPDATE_CELL

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

        // compute next coordinates
        cell.coords += cell.exit_step * u_ray.signs;

        // update stats
        #if DEBUG_ENABLED == 1

            stats.num_cells += 1;

        #endif

        // UPDATE_CUBIC     
        vec3 span_position = cell.exit_position - cell.entry_position;

        cubic.values.x = cubic.values.w;
        cubic.values.y = sampleVolume(cell.entry_position + span_position * (1.0 / 3.0));
        cubic.values.z = sampleVolume(cell.entry_position + span_position * (2.0 / 3.0));
        cubic.values.w = sampleVolume(cell.exit_position);

        #if DEBUG_ENABLED == 1

            stats.num_volume_fetches += 3;

        #endif

        // BERNSTEIN_TEST
        cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;

        if (any(greaterThan(cubic.bernstein_coeffs, vec4(mip.value))))
        {
            // SOLVE_CUBIC
            cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
            CubicMax cubicMax = cubicMaxFromCoeffs_v2(cubic.coeffs);
            
            cubic.argmax_time = cubicMax.t;
            cubic.max_value = cubicMax.v;
            
            #if DEBUG_ENABLED == 1

                stats.num_cubics += 1;

            #endif

            // UPDATE_MIP
            mip.distance = cell.entry_distance + cell.span_distance * cubic.argmax_time;
            mip.value = cubic.max_value;

            #if DEBUG_ENABLED == 1

                stats.num_mips += 1;

            #endif
        }

        if (cell.exit_distance > block.exit_distance || cell.terminated) break; 
    }

    if (cell.terminated) break;
}

// END_MIP
mip.position = rayDistanceToPosition(mip.distance); 
mip.gradient = compute_gradient(mip.position, mip.hessian);
mip.curvatures = compute_curvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);


