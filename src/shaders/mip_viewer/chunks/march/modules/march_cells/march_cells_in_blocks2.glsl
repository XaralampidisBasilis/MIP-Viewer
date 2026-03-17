
// START_CUBIC_CELL
cell.exit_distance = ray.start_distance;
cell.exit_position = rayDistanceToPosition(cell.exit_distance); 

cell.coords = positionToCellCoords(cell.exit_position);
cubic.values[3] = sampleVolume(cell.exit_position);

// START_MIP
mip.value = cubic.values[3];
mip.distance = ray.start_distance;

#if DEBUG_ENABLED == 1

    stats.num_mips += 1;
    stats.num_fetches += 1;
    stats.num_volume_fetches += 1;
    
#endif

for (int i = 0; i < MAX_CELLS; i++) 
{
    // UPDATE_SKIP_CELL

    // compute skip distance
    cell.skip_distance = sample_rgba16ui_distance_fast(cell.coords, cell.shadowed);
    // cell.skip_distance = sample_rgb32ui_distance_fast(cell.coords, cell.shadowed);

    bool single = (cell.skip_distance == 1);

    // compute entry from previous exit
    cell.entry_distance = cell.exit_distance;
    cell.entry_position = cell.exit_position;

    // compute exit from cell ray intersection 
    float nudge = (single) ? 0.0 : ray.spacing * 0.001;
    cell.exit_distance = intersectSkipCellExit(cell.coords, cell.skip_distance, cell.exit_normal) + nudge;
    cell.exit_position = rayDistanceToPosition(cell.exit_distance);

    // compute span distance
    cell.span_distance = cell.exit_distance - cell.entry_distance;

    // compute termination condition
    cell.terminated = cell.exit_distance > ray.end_distance;

    // compute next coordinates
    if (single)
    {
        cell.coords += cell.exit_normal * u_ray.signs;
    }
    else
    {
        ivec3 exit_coords = positionToCellCoords(cell.exit_position);
        ivec3 skip_coords = cell.coords + cell.skip_distance * u_ray.signs;
        cell.coords = mmix(exit_coords, skip_coords, cell.exit_normal);
    }

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_fetches += 1;
        stats.num_cells += 1;

    #endif

    if (!cell.terminated && cell.shadowed) continue;

    // UPDATE_CUBIC
    cell.exit_distance = clamp(cell.exit_distance, ray.start_distance, ray.end_distance);

    // compute cubic values
    cubic.values[0] = sampleVolume(mix(cell.entry_position, cell.exit_position, 0.0 / 3.0));
    cubic.values[1] = sampleVolume(mix(cell.entry_position, cell.exit_position, 1.0 / 3.0));
    cubic.values[2] = sampleVolume(mix(cell.entry_position, cell.exit_position, 2.0 / 3.0));
    cubic.values[3] = sampleVolume(mix(cell.entry_position, cell.exit_position, 3.0 / 3.0));

    // compute cubic coefficients
    cubic.coeffs = cubic.values * cubic_inv_vander;

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_volume_fetches += 1;
        stats.num_fetches += 4;

    #endif

    // UPDATE_MIP

    vec4 c = cubic.coeffs;
    vec2 p = quadratic_roots(vec3(c.y, 2.0 * c.z, 3.0 * c.w));
    p = clamp(p, 0.0, 1.0);

    vec2 v_ext = eval_poly(c, p);
    vec4 v = vec4(cubic.values[0], v_ext.x, v_ext.y, cubic.values[3]);
    vec4 t = vec4(0.0, p.x, p.y, 1.0);

    int i_max = argmax(v);
    float v_max = v[i_max];
    float t_max = p[i_max];

    if (v_max > mip.value) 
    {
        mip.value = v_max;
        mip.distance = mix(cell.entry_distance, cell.exit_distance, t_max);

        #if DEBUG_ENABLED == 1

            stats.num_mips += 1;

        #endif
    }

    if (cell.terminated) break;
}

#include "./end_mip"

