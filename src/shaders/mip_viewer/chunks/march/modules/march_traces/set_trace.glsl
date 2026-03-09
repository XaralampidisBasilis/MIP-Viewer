
trace.spacing = ray.spacing / 3.0;

float jitter = random(ray.start_position);
trace.distance = trace.spacing * (floor(ray.start_distance / trace.spacing) + jitter);

trace.distance = clamp(trace.distance, ray.start_distance, ray.end_distance);
trace.position = camera.position + ray.direction * trace.distance; 

trace.value = sample_volume(trace.position);

#if DEBUG_ENABLED == 1

    stats.num_traces += 1;
    stats.num_fetches += 1; 
    stats.volume_samples += 1;

#endif

