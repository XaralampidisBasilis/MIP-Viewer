
// start value
trace.value = sample_volume(trace.position);

// update stats
#if DEBUG_ENABLED == 1

    stats.num_fetches += 1;
    stats.volume_samples += 1;

#endif

