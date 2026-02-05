
trace.spacing = ray.spacing;

// start trace
#if SKIPPING_ENABLED == 1
trace.distance = block.entry_distance;
trace.distance -= trace.spacing * random(block.entry_position);

#else
trace.distance = ray.start_distance;
trace.distance -= trace.spacing * random(ray.start_position);
#endif

// start interpolant
trace.position = camera.position + ray.direction * trace.distance; 
trace.value = sample_volume(trace.position);

#if DEBUG_ENABLED == 1
stats.num_fetches += 1;
#endif

