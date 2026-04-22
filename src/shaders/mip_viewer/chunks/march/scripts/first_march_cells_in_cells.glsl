
// start block at ray start

// START_CELL_IN_RAY
cell.coords = positionToCellCoords(ray.start_position);
cell.far_distances = cellCoordsToFarDistances(cell.coords);

cell.exit_distance = ray.start_distance;
cell.exit_position = ray.start_position; 
cell.exit_step = ivec3(0);

// start cubic at the ray start

// START_CUBIC_IN_RAY
cubic.values.w = sampleVolume(ray.start_position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;

#endif

// start mip at the ray start

// START_MIP_IN_CUBIC
mip.update = shouldUpdateMip(mip.value, cubic.max_value);

if (mip.update)
{
    mip.distance = ray.start_distance;
    mip.value = cubic.max_value;

    #if DEBUG_ENABLED == 1

        stats.num_mips += 1;

    #endif

}

// START_MARCH
for (int i = 0; i < MAX_CELLS; i++) 
{
    // update cell based on the previous one
    // UPDATE_CELL_IN_RAY
    
    // compute far distances
    cell.coords = advanceCellCoords(cell.coords, cell.exit_step);
    cell.far_distances = advanceCellFarDistances(cell.far_distances, cell.exit_step);
    
    // compute entry from previous exit
    cell.entry_distance = cell.exit_distance;
    cell.entry_position = cell.exit_position;
    
    // compute exit from cell ray intersection 
    cell.exit_distance = intersectCellFarDistances(cell.far_distances, cell.exit_step);
    cell.exit_position = distanceToPosition(cell.exit_distance);
    
    // compute span distance
    cell.span_distance = cell.exit_distance - cell.entry_distance;
    
    // compute termination condition
    cell.terminated = cell.exit_distance > ray.end_distance - ray.eps_distance;
    if (cell.terminated)
    {
        cell.exit_distance = ray.end_distance;
        cell.exit_position = ray.end_position;
    }
    
    // update stats
    #if DEBUG_ENABLED == 1
    
        stats.num_cells += 1;
    
    #endif

    // Reconstruct the cubic polynomial inside the cell entry and exit
    
    // UPDATE_CUBIC_IN_CELL
    vec3 span_vector = cell.exit_position - cell.entry_position;
    
    cubic.values.x = cubic.values.w;
    cubic.values.y = sampleVolume(cell.entry_position + span_vector * (1.0 / 3.0));
    cubic.values.z = sampleVolume(cell.entry_position + span_vector * (2.0 / 3.0));
    cubic.values.w = sampleVolume(cell.exit_position);
    
    #if DEBUG_ENABLED == 1
    
        stats.num_volume_fetches += 3;
    
    #endif

    // Maximize the cubic inside the cell 
    
    #if BERNSTEIN_ENABLED == 1
    
    // Cull with Bernstein coefficients before the full cubic maximize step.
    cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;
    cubic.maximize = any(greaterThan(cubic.bernstein_coeffs, vec4(mip.value)));
    
    if (cubic.maximize)
    {
        // MAXIMIZE_CUBIC
        cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
        CubicMax cubic_max = cubicMaxOnUnitInterval(cubic.coeffs, cubic.values.x, cubic.values.w);
    
        cubic.max_value = cubic_max.value;
        cubic.argmax_point = cubic_max.point;
    
        #if DEBUG_ENABLED == 1
    
            stats.num_maxima += 1;
    
        #endif
    }

    #else
    
    // MAXIMIZE_CUBIC
    cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
    CubicMax cubic_max = cubicMaxOnUnitInterval(cubic.coeffs, cubic.values.x, cubic.values.w);
    
    cubic.maximize = true;
    cubic.max_value = cubic_max.value;
    cubic.argmax_point = cubic_max.point;
    
    #if DEBUG_ENABLED == 1
    
        stats.num_maxima += 1;
    
    #endif
    
    #endif
    

    // Update mip based on the max cubic value
    
    
    // UPDATE_MIP_IN_CUBIC
    mip.update = shouldUpdateMip(mip.value, cubic.max_value) && cubic.maximize;
    
    if (mip.update) 
    {
        mip.distance = mix(cell.entry_distance, cell.exit_distance, cubic.argmax_point);
        mip.value = cubic.max_value;
    
        #if DEBUG_ENABLED == 1
    
            stats.num_mips += 1;
    
        #endif
    }

    if (cell.terminated) break; 
}

// END_RAY_IN_MIP

// END_RAY_IN_MIP
ray.end_distance = mip.distance + ray.eps_distance;
ray.end_position = distanceToPosition(ray.end_distance);

ray.span_distance = ray.end_distance - ray.start_distance;




