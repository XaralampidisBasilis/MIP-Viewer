
// set spacing
trace.step_distance = ray.step_distance / 2.0;

// set distance with phase
trace.distance = ray.start_distance + trace.step_distance * ray.phase;

// set position
trace.position = distanceToPosition(trace.distance); 

// set value
trace.value = sampleVolume(trace.position);

// update stats
#if DEBUG_ENABLED == 1

    stats.num_traces += 1;
    stats.num_volume_fetches += 1;

#endif

