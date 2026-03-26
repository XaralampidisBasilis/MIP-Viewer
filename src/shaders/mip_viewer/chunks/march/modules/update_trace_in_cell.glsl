
// point inside the cell entry and exit
float ti = float(i) / float(MAX_TRACES_IN_CELL - 1);

// Increment distance
trace.distance = mix(cell.entry_distance, cell.exit_distance, ti);

// Compute position
trace.position = distanceToPosition(trace.distance); 

// Update value
trace.value = sampleVolume(trace.position);

// Compute termination condition
trace.terminated = trace.distance > ray.end_distance; 

// update stats
#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;
    stats.num_traces += 1;

#endif