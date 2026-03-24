

float ray_end_distance = ray.end_distance;

// START_BLOCK
block.coords = positionToBlockCoords(ray.start_position);
block.exit_position = ray.start_position; 
block.exit_step = ivec3(0);
block.empty = false;

// START_CUBIC
cubic.values.w = sampleVolume(ray.start_position);

// START_MIP
mip.value = cubic.values.w;

// START_MARCH

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    // UPDATE_BLOCK

    // Choose next block coords from either geometric exit or skip step
    block.coords = advanceBlockCoords(block.coords, block.exit_step, block.step_radius, block.exit_position + ray.eps_direction);

    // Read skip radius and shadow flag for the current block
    block.prev_empty = block.empty;
    // block.step_radius = sampleDistance1bit(block.coords, block.empty);
    // block.step_radius = sampleDistance5bit(block.coords, block.empty);
    block.step_radius = sampleDistance8bit(block.coords, block.empty);

    // Current entry is the previous step's exit
    block.entry_position = block.exit_position;
    block.entry_step = block.exit_step;

    // Find exit point of the current skip block
    block.exit_distance = intersectBlockExit(block.coords, block.step_radius, block.exit_step);
    block.exit_position = distanceToPosition(block.exit_distance);

    // Stop once the ray exit goes beyond the ray end
    block.terminated = block.exit_distance > ray_end_distance;

    if (block.empty)
    {
        if (!block.terminated) continue; else break;    
    }

    // START_CELL_TO_BLOCK
    cell.coords = startCellCoordsInBlock(block.coords, block.entry_step, block.entry_position + ray.eps_direction);
    cell.exit_position = block.entry_position; 
    cell.exit_step = ivec3(0);

    // START_CUBIC
    if(block.prev_empty)
    {
        cubic.values.w = sampleVolume(block.entry_position);
    }
      
    // START_MARCH_IN_BLOCK
    #pragma unroll
    for (int i = 0; i < MAX_CELLS_IN_BLOCK; i++)
    {
        // UPDATE_CELL

        // compute next coordinates
        cell.coords = advanceCellCoords(cell.coords, cell.exit_step);

        // compute entry from previous exit
        cell.entry_position = cell.exit_position;

        // compute exit from cell ray intersection 
        cell.exit_distance = intersectCellExit(cell.coords, cell.exit_step);
        cell.exit_position = distanceToPosition(cell.exit_distance);

        // compute termination condition
        cell.terminated = 
            cell.exit_distance > block.exit_distance - ray.eps_distance || 
            cell.exit_distance > ray_end_distance;

        // UPDATE_CUBIC     
        vec3 span_vector = cell.exit_position - cell.entry_position;

        cubic.values.x = cubic.values.w;
        cubic.values.y = sampleVolume(cell.entry_position + span_vector * (1.0 / 3.0));
        cubic.values.z = sampleVolume(cell.entry_position + span_vector * (2.0 / 3.0));
        cubic.values.w = sampleVolume(cell.exit_position);

        // BERNSTEIN_TEST
        vec4 bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;
        bool mip_update = 
            bernstein_coeffs.x > mip.value || 
            bernstein_coeffs.y > mip.value || 
            bernstein_coeffs.z > mip.value || 
            bernstein_coeffs.w > mip.value;

        if (mip_update)
        {
            // SOLVE_CUBIC
            vec4 power_coeffs = cubic.values * CUBIC_INV_VANDER;
            CubicMax cubic_max = cubicMaxOnUnitInterval(power_coeffs, cubic.values.x, cubic.values.w);

            // UPDATE_MIP
            mip.value = max(mip.value, cubic_max.v);
        }

        if (cell.terminated) break; 
    }

    if (block.terminated) break;
}
