
// start distance with phase from block
trace.distance = snapToTraceDistance(block.entry_distance, trace.step_distance, ray.phase);

// update position
trace.position = distanceToPosition(trace.distance); 

// start value
trace.value = sampleVolume(trace.position);

// update stats
#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;
    stats.num_traces += 1;

#endif

