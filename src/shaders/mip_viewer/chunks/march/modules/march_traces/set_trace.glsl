
// set spacing
trace.spacing = ray.spacing / 2.0;

// set distance with jitter
float jitter = random(ray.start_position);
trace.distance = trace.spacing * (ceil(ray.start_distance / trace.spacing) + jitter);

// set position
trace.position = camera.position + ray.direction * trace.distance; 

// set value
trace.value = sample_volume(trace.position);

// update stats
#if DEBUG_ENABLED == 1

    stats.num_traces += 1;
    stats.num_fetches += 1; 
    stats.volume_samples += 1;

#endif

