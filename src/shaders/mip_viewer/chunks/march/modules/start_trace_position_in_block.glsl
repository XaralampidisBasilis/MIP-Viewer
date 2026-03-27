
// START_TRACE_IN_BLOCK
trace.distance = snapTraceDistanceFloor(block.entry_distance, trace.step_distance, ray.phase);

// Compute position
trace.position = distanceToPosition(trace.distance); 

// update stats
#if DEBUG_ENABLED == 1

    stats.num_traces += 1;

#endif