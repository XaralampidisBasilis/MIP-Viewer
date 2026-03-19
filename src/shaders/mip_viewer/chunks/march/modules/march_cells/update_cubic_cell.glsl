
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

// compute cubic values
cubic.values[0] = cubic.values[3];
cubic.values[1] = sampleVolume(mix(cell.entry_position, cell.exit_position, 1.0 / 3.0));
cubic.values[2] = sampleVolume(mix(cell.entry_position, cell.exit_position, 2.0 / 3.0));
cubic.values[3] = sampleVolume(mix(cell.entry_position, cell.exit_position, 3.0 / 3.0));

// compute cubic coefficients
cubic.coeffs = cubic.values * CUBIC_INV_VANDER;

// update stats
#if DEBUG_ENABLED == 1

    stats.num_cells += 1;
    stats.num_volume_fetches += 3;

#endif