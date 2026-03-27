
// Increment distance
trace.distance += trace.step_distance;

// Compute position
trace.position = distanceToPosition(trace.distance); 

// Compute termination condition
trace.terminated = trace.distance > block.exit_distance; 

// update stats
#if DEBUG_ENABLED == 1

    stats.num_traces += 1;

#endif