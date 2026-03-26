
// Increment distance
trace.distance = mix(cell.entry_distance, cell.exit_distance, float(i));

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