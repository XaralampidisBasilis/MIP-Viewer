
// Update value
trace.value = sample_volume(trace.position);

// update stats
#if DEBUG_ENABLED == 1

    stats.num_fetches += 1;
    stats.num_volume_fetches += 1;

#endif