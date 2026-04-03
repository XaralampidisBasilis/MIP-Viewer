
// START_CUBIC_IN_RAY
cubic.values.w = sampleVolume(ray.start_position);

cubic.max_value = cubic.values.w;
cubic.argmax_point = ray.start_distance;

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;

#endif