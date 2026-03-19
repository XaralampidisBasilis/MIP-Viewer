
// start distance with phase from block
// trace.distance = trace.spacing * (ceil(block.entry_distance / trace.spacing) + ray.phase);
trace.distance = block.entry_distance + trace.spacing * ray.phase;

// clamp distance
trace.distance = clamp(trace.distance, ray.start_distance, ray.end_distance);

// update position
trace.position = rayDistanceToPosition(trace.distance); 

// start value
trace.value = sampleVolume(trace.position);

// update stats
#if DEBUG_ENABLED == 1

    stats.num_traces += 1;
    stats.num_volume_fetches += 1;

#endif

