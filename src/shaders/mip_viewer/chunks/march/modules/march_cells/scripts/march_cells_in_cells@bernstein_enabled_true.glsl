

// START_CELL
cell.coords = positionToCellCoords(ray.start_position);
cell.exit_distance = ray.start_distance;
cell.exit_position = ray.start_position; 
cell.exit_step = ivec3(0);

// START_CUBIC
cubic.values.w = sampleVolume(ray.start_position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;

#endif

// START_MIP
mip.distance = ray.start_distance;
mip.value = cubic.values.w;

// update stats
#if DEBUG_ENABLED == 1

    stats.num_mips += 1;

#endif

// START_MARCH

for (int i = 0; i < MAX_CELLS; i++) 
{
    // UPDATE_CELL

    // compute coordinates
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
    cell.terminated = cell.exit_distance > ray.end_distance;

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

    // MAXIMIZE_BERNSTEIN
    cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;
    mip.update = any(greaterThan(cubic.bernstein_coeffs, vec4(mip.value)));

    if (mip.update)
    {
        // MAXIMIZE_CUBIC
        cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
        CubicMax cubic_max = cubicMaxOnUnitInterval(cubic.coeffs, cubic.values.x, cubic.values.w);
        
        cubic.argmax_point = cubic_max.point;
        cubic.max_value = cubic_max.value;

        #if DEBUG_ENABLED == 1

            stats.num_maxima += 1;

        #endif

        mip.update = cubic.max_value > mip.value;
    }

    // UPDATE_MIP
    if (mip.update)
    {
        mip.distance = mix(block.entry_distance, block.exit_distance, cubic.argmax_point);
        mip.value = cubic.max_value;

        #if DEBUG_ENABLED == 1

            stats.num_mips += 1;

        #endif
    }
        
    if (cell.terminated) break;
}

// END_MIP
mip.terminated = mip.distance > ray.end_distance;
mip.position = distanceToPosition(mip.distance); 
mip.gradient = computeGradient(mip.position, mip.hessian);
mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);






