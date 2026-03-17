
// set spacing
trace.spacing = ray.spacing / 2.0;

// set distance with phase
trace.distance = ray.start_distance + trace.spacing * ray.phase;

// set position
trace.position = rayDistanceToPosition(trace.distance); 

// set value
trace.value = sampleVolume(trace.position);

// update stats
#if DEBUG_ENABLED == 1

    stats.num_traces += 1;
    stats.num_fetches += 1; 
    stats.num_volume_fetches += 1;

#endif

