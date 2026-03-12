
// start distance with jitter from block
float jitter = random(block.entry_position);
trace.distance = trace.spacing * (floor(block.entry_distance / trace.spacing) + jitter);

// clamp distance
trace.distance = clamp(trace.distance, ray.start_distance, ray.end_distance);

// update position
trace.position = camera.position + ray.direction * trace.distance; 

// start value
trace.value = sample_volume(trace.position);

// update stats
#if DEBUG_ENABLED == 1

    stats.num_traces += 1;
    stats.num_fetches += 1;
    stats.num_volume_fetches += 1;

#endif

