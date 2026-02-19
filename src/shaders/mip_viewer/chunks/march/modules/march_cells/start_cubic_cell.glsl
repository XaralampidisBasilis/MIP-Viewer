
// start cell
#if SKIPPING_ENABLED == 1

    cell.exit_distance = block.entry_distance;
    cell.exit_position = block.entry_position;

#else

    cell.exit_distance = ray.start_distance;
    cell.exit_position = ray.start_position;

#endif

cell.coords = ivec3(round(cell.exit_position)); 

cubic.values[3] = sample_volume(cell.exit_position);

#if DEBUG_ENABLED == 1

    stats.num_fetches += 1;
    
#endif

