
// START_TRACE_IN_RAY

// set spacing
trace.step_distance = ray.step_distance / float(MAX_TRACES_IN_CELL);

// set position
trace.distance = snapTraceDistanceCeil(ray.start_distance, trace.step_distance, ray.phase);
trace.position = distanceToPosition(trace.distance); 

// set value
trace.value = sampleVolume(trace.position);

// update stats
#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;
    stats.num_traces += 1;

#endif

