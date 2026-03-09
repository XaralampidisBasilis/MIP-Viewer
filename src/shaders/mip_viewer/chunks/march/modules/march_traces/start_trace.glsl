
trace.spacing = ray.spacing / 2.0;

// start trace
#if SKIPPING_ENABLED == 1

    trace.distance = floor(block.entry_distance / trace.spacing) * trace.spacing;
    trace.distance += trace.spacing * random(block.entry_position);
    trace.distance = clamp(trace.distance, ray.start_distance, ray.end_distance);

#else

    trace.distance = floor(ray.start_distance / trace.spacing) * trace.spacing;
    trace.distance += trace.spacing * random(ray.start_position);

#endif

// start interpolant
trace.position = camera.position + ray.direction * trace.distance; 
trace.value = sample_volume(trace.position);

#if DEBUG_ENABLED == 1

    stats.num_traces += 1;
    stats.num_fetches += 1;
    stats.volume_samples += 1;

#endif

