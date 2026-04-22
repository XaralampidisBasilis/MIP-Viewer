
// START_TRACE_IN_RAY

// START_TRACE_IN_RAY

// set spacing
trace.step_distance = u_ray.step_distance / float(MAX_TRACES_IN_CELL - 1);

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


// START_MIP_IN_TRACE

// START_MIP_IN_TRACE
mip.update = shouldUpdateMip(mip.value, trace.value);

if (mip.update)
{
    mip.distance = trace.distance;
    mip.value = trace.value;

    #if DEBUG_ENABLED == 1

        stats.num_mips += 1;

    #endif

}

// MARCH_TRACES
for (int i = 0; i < MAX_TRACES; i++) 
{
    // UPDATE_TRACE_POSITION_IN_RAY
    
    // Increment distance
    trace.distance += trace.step_distance;
    
    // Compute position
    trace.position = distanceToPosition(trace.distance); 
    
    // Compute termination condition
    trace.terminated = trace.distance > ray.end_distance; 
    
    // update stats
    #if DEBUG_ENABLED == 1
    
        stats.num_traces += 1;
    
    #endif

    // TERMINATE_MARCH_TRACES
    if (trace.terminated) break; 

    // UPDATE_TRACE_VALUE
    
    // Update value
    trace.value = sampleVolume(trace.position);
    
    // update stats
    #if DEBUG_ENABLED == 1
    
        stats.num_volume_fetches += 1;
    
    #endif

    // UPDATE_MIP_IN_TRACE
    
    // UPDATE_MIP_IN_TRACE
    mip.update = shouldUpdateMip(mip.value, trace.value);
    
    if (mip.update)
    {
        mip.distance = trace.distance;
        mip.value = trace.value;
    
        #if DEBUG_ENABLED == 1
    
            stats.num_mips += 1;
    
        #endif
    
    }
    
}

// END_RAY_IN_MIP

// END_RAY_IN_MIP
ray.end_distance = mip.distance + ray.eps_distance;
ray.end_position = distanceToPosition(ray.end_distance);

ray.span_distance = ray.end_distance - ray.start_distance;



