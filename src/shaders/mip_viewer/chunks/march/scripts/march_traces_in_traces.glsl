
// START_TRACE_IN_RAY
trace.step_distance = ray.step_distance / float(MAX_TRACES_IN_CELL - 1);

trace.distance = snapTraceDistanceCeil(ray.start_distance, trace.step_distance, ray.phase);
trace.position = distanceToPosition(trace.distance); 

trace.value = sampleVolume(trace.position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;
    stats.num_traces += 1;

#endif

// START_MIP_IN_TRACE
mip.distance = trace.distance;
mip.value = trace.value;

#if DEBUG_ENABLED == 1

    stats.num_mips += 1;

#endif

// MARCH_TRACES
for (int i = 0; i < MAX_TRACES; i++) 
{
    // UPDATE_TRACE_POSITION_IN_RAY
    trace.distance += trace.step_distance;
    trace.position = distanceToPosition(trace.distance); 

    trace.terminated = trace.distance > ray.end_distance; 

    #if DEBUG_ENABLED == 1

        stats.num_traces += 1;

    #endif

    if (trace.terminated) break; 

    // UPDATE_TRACE_VALUE
    trace.value = sampleVolume(trace.position);

    #if DEBUG_ENABLED == 1

        stats.num_volume_fetches += 1;

    #endif

    // UPDATE_MIP_IN_TRACE
    mip.update = trace.value > mip.value;

    if (mip.update)
    {
        mip.distance = trace.distance;
        mip.value = trace.value;

        #if DEBUG_ENABLED == 1

            stats.num_mips += 1;

        #endif
    }
}

// END_MIP
mip.terminated = mip.distance > ray.end_distance;
mip.position = distanceToPosition(mip.distance); 

mip.gradient = computeGradient(mip.position, mip.hessian);
mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);



