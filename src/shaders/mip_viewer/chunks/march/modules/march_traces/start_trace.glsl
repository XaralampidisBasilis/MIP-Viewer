
trace.spacing = ray.spacing / 3.0;

// start trace
#if SKIPPING_ENABLED == 1

    trace.distance = block.entry_distance;

#else

    trace.distance = ray.start_distance;
    
#endif

// start interpolant
trace.position = camera.position + ray.direction * trace.distance; 
trace.value = sample_volume(trace.position);

#if DEBUG_ENABLED == 1

    stats.num_traces += 1;
    stats.num_fetches += 1;

#endif

