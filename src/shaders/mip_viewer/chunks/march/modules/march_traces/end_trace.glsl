
trace.distance = ray.end_distance;
trace.position = camera.position + ray.direction * trace.distance; 
trace.value = sample_volume(trace.position);

#if DEBUG_ENABLED == 1

    stats.num_traces += 1;
    stats.num_fetches += 1;
    stats.volume_samples += 1;

#endif

