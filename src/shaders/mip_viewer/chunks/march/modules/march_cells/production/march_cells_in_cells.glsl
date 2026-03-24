

// START_CELL_IN_RAY
cell.coords = positionToCellCoords(ray.start_position);
cell.far_distances = cellCoordsToFarDistances(cell.coords);
cell.exit_position = ray.start_position; 
cell.exit_step = ivec3(0);

// START_CUBIC_IN_RAY
cubic.values.w = sampleVolume(ray.start_position);

// START_MIP_IN_CUBIC
mip.value = cubic.values.w;

// START_MARCH
for (int i = 0; i < MAX_CELLS; i++) 
{
    // UPDATE_CELL_IN_RAY

    // compute coordinates
    cell.far_distances = advanceCellFarDistances(cell.far_distances, cell.exit_step);
    cell.coords = advanceCellCoords(cell.coords, cell.exit_step);

    // compute entry from previous exit
    cell.entry_position = cell.exit_position;

    // compute exit from cell ray intersection 
    cell.exit_distance = intersectCellFarDistances(cell.far_distances, cell.exit_step);
    cell.exit_position = distanceToPosition(cell.exit_distance);

    // compute termination condition
    cell.terminated = cell.exit_distance > ray.end_distance;

    // UPDATE_CUBIC     
    vec3 span_vector = cell.exit_position - cell.entry_position;

    cubic.values.x = cubic.values.w;
    cubic.values.y = sampleVolume(cell.entry_position + span_vector * (1.0 / 3.0));
    cubic.values.z = sampleVolume(cell.entry_position + span_vector * (2.0 / 3.0));
    cubic.values.w = sampleVolume(cell.exit_position);

    // MAXIMIZE_BERNSTEIN_IN_CELL
    vec4 bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;
    bool mip_update = 
        bernstein_coeffs.x > mip.value || 
        bernstein_coeffs.y > mip.value || 
        bernstein_coeffs.z > mip.value || 
        bernstein_coeffs.w > mip.value;

    if (mip_update)
    {
        // MAXIMIZE_CUBIC_IN_CELL
        vec4 coeffs = cubic.values * CUBIC_INV_VANDER;
        CubicMax cubic_max = cubicMaxOnUnitInterval(coeffs, cubic.values.x, cubic.values.w);

        // UPDATE_MIP_IN_CUBIC
        mip.value = max(mip.value, cubic_max.value);
    }
        
    if (cell.terminated) break;
}






