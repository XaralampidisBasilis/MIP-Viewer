
// compute the intensity samples inside the cell from the intensity map texture
cubic.values[0] = cubic.values[3];

#pragma unroll
for (int i = 1; i < 4; i++) 
{
    vec3 position = mix(cell.entry_position, cell.exit_position, sampling_points[i]);

    cubic.values[i] = sample_volume(position);
}

cubic.coeffs = cubic.values * cubic_inv_vander;

#if DEBUG_ENABLED == 1

    stats.num_fetches += 3;
    
#endif
