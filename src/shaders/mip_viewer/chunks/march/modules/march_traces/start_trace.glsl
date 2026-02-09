
trace.spacing = ray.spacing / 3.0;

// start trace
#if SKIPPING_ENABLED == 1

    float jitter = random(block.entry_position);
    trace.distance = block.entry_distance;
    // trace.distance -= ray.spacing * jitter;

#else

    float jitter = random(ray.start_position);
    trace.distance = ray.start_distance;
    // trace.distance -= ray.spacing * jitter;
    
#endif

// start interpolant
trace.position = camera.position + ray.direction * trace.distance; 
trace.value = sample_volume(trace.position);

#if DEBUG_ENABLED == 1

    stats.num_fetches += 1;

#endif

