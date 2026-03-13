
// start cell
#if SKIPPING_ENABLED == 1

    cell.exit_distance = block.entry_distance;

#else

    cell.exit_distance = ray.start_distance;

#endif

cell.exit_distance = clamp(cell.exit_distance, ray.start_distance, ray.end_distance);
cell.exit_position = ray.origin + ray.direction * cell.exit_distance; 
cell.coords = ivec3(round(cell.exit_position)); 

cubic.values[3] = sample_volume(cell.exit_position);

#if DEBUG_ENABLED == 1

    stats.num_fetches += 1;
    
#endif

