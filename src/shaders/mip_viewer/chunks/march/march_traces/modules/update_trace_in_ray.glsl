

// Increment distance
trace.distance += trace.step_distance;

// Compute position
trace.position = distanceToPosition(trace.distance); 

// Compute termination condition
trace.terminated = trace.distance > ray.end_distance; 

// Update value
trace.value = sampleVolume(trace.position);

// update stats
#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;
    stats.num_traces += 1;

#endif