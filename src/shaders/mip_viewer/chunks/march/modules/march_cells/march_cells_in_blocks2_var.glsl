
float eps_distance = u_ray.spacing * 0.001;
vec3 eps_direction = u_ray.direction * eps_distance;

// START_BLOCK
block.coords = positionToBlockCoords(ray.start_position + eps_direction);
block.exit_distance = ray.start_distance;
block.exit_position = ray.start_position; 
block.exit_step = ivec3(0);
block.empty = false;

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

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    // UPDATE_BLOCK

    // Choose next block coords from either geometric exit or skip step
    block.coords = advanceBlockCoords(block.coords, block.exit_step, block.step_radius, block.exit_position + eps_direction);

    // Read skip radius and shadow flag for the current block
    bool prev_empty = block.empty;
    // block.step_radius = sampleDistance5bit(block.coords, block.empty);
    block.step_radius = sampleDistance8bit(block.coords, block.empty);

    // Current entry is the previous step's exit
    block.entry_distance = block.exit_distance;
    block.entry_position = block.exit_position;
    block.entry_step = block.exit_step;

    // Find exit point of the current skip block
    block.exit_distance = intersectBlockExit(block.coords, block.step_radius, block.exit_step);
    block.exit_position = distanceToPosition(block.exit_distance);

    // Distance covered inside this block span
    block.span_distance = block.exit_distance - block.entry_distance;

    // Stop once the ray exit goes beyond the ray end
    block.terminated = block.exit_distance > ray.end_distance - eps_distance;

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_blocks += 1;

    #endif

    if (block.empty)
    {
        if (!block.terminated) continue; else break;    
    }

    // START_CELL_TO_BLOCK
    cell.coords = advanceCellCoordsAtBlock(block.coords, block.entry_step, block.entry_position + eps_direction);
    cell.exit_distance = block.entry_distance;
    cell.exit_position = block.entry_position; 
    cell.exit_step = ivec3(0);

    // START_CUBIC
    if (prev_empty)
    {
        cubic.values.w = sampleVolume(block.entry_position);

        #if DEBUG_ENABLED == 1

            stats.num_volume_fetches += 1;
        
        #endif
    }
      
    // START_MARCH_IN_BLOCK
    #pragma unroll
    for (int i = 0; i < MAX_CELLS_IN_BLOCK; i++)
    {
        // UPDATE_CELL

        // compute next coordinates
        cell.coords = advanceCellCoords(cell.coords, cell.exit_step);

        // compute entry from previous exit
        cell.entry_distance = cell.exit_distance;
        cell.entry_position = cell.exit_position;

        // compute exit from cell ray intersection 
        cell.exit_distance = intersectCellExit(cell.coords, cell.exit_step);
        cell.exit_position = distanceToPosition(cell.exit_distance);

        // compute span distance
        cell.span_distance = cell.exit_distance - cell.entry_distance;

        // compute termination condition
        cell.terminated = 
            cell.exit_distance > block.exit_distance - eps_distance || 
            cell.exit_distance > ray.end_distance - eps_distance;

        // update stats
        #if DEBUG_ENABLED == 1

            stats.num_cells += 1;

        #endif

        // UPDATE_CUBIC     
        vec3 span_vector = cell.exit_position - cell.entry_position;

        cubic.values.x = cubic.values.w;
        cubic.values.y = sampleVolume(cell.entry_position + span_vector * (1.0 / 3.0));
        cubic.values.z = sampleVolume(cell.entry_position + span_vector * (2.0 / 3.0));
        cubic.values.w = sampleVolume(cell.exit_position);

        #if DEBUG_ENABLED == 1

            stats.num_volume_fetches += 3;

        #endif

        // BERNSTEIN_TEST
        cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;
        mip.update = any(greaterThan(cubic.bernstein_coeffs, vec4(mip.value)));

        if (mip.update)
        {
            // SOLVE_CUBIC
            cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
            CubicMax cubic_max = cubicMaxOnUnitInterval(cubic.coeffs, cubic.values.x, cubic.values.w);
            
            cubic.argmax_time = cubic_max.t;
            cubic.max_value = cubic_max.v;
            
            #if DEBUG_ENABLED == 1

                stats.num_cubics += 1;

            #endif

            // UPDATE_MIP
            mip.distance = mix(cell.entry_distance, cell.exit_distance, cubic_max.t);
            mip.value = cubic_max.v;

            #if DEBUG_ENABLED == 1

                stats.num_mips += 1;

            #endif
        }

        if (cell.terminated) break; 
    }

    if (block.terminated) break;
}

// END_MIP
mip.terminated = mip.distance > ray.end_distance - eps_distance;

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


