
// compute box min/max positions
cell.min_position = vec3(cell.coords) - 0.5;
cell.max_position = vec3(cell.coords) + 0.5;

// compute entry from previous exit
cell.entry_distance = cell.exit_distance;
cell.entry_position = cell.exit_position;

// compute exit from cell ray intersection 
cell.exit_distance = intersect_box_exit(cell.min_position, cell.max_position, ray.origin, ray.inv_direction, cell.exit_normal);
cell.exit_position = distanceToPosition(cell.exit_distance); 

// compute span distance
cell.span_distance = cell.exit_distance - cell.entry_distance;

// compute cubic values
cubic.values[0] = cubic.values[3];

const vec4 sampling_points = vec4(0, 1, 2, 3) / 3.0;
#pragma unroll
for (int i = 1; i <= 3; i++) 
{
    vec3 position = mix(cell.entry_position, cell.exit_position, sampling_points[i]);
    cubic.values[i] = sample_volume(position);
}

// compute cubic coefficients
cubic.coeffs = cubic.values * cubic_inv_vander;

// compute next coordinates
cell.coords += cell.exit_normal * ray.signs;

// compute termination condition
cell.terminated = cell.exit_distance > ray.end_distance; 

// update stats
#if DEBUG_ENABLED == 1

    stats.num_cells += 1;
    stats.num_fetches += 3;

#endif