
#include "../march_blocks/start_block"

#include "./start_cubic_cell"
#include "./start_mip"

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    #include "../march_blocks/update_block"

    if (block.shadowed && !block.terminated) continue;
        
    // compute cubic values
    cubic.values[0] = cubic.values[3];

    const vec4 sampling_points = vec4(0, 1, 2, 3) / 3.0;
    #pragma unroll
    for (int i = 1; i <= 3; i++) 
    {
        vec3 position = mix(block.entry_position, block.exit_position, sampling_points[i]);
        cubic.values[i] = sampleVolume(position);
    }

    // compute cubic coefficients
    cubic.coeffs = cubic.values * cubic_inv_vander;

    // compute next coordinates
    cell.coords += block.exit_normal * ray.signs;

    // compute termination condition
    cell.terminated = block.exit_distance > ray.end_distance; 

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_cells += 1;
        stats.num_fetches += 3;

    #endif

    #include "./update_mip"


    if (cell.terminated) break;
}

#include "./end_mip"

