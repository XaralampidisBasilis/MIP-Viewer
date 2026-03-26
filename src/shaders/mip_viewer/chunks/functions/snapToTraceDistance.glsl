#ifndef SNAP_TO_TRACE_DISTANCE
#define SNAP_TO_TRACE_DISTANCE

float snapToTraceDistance(float t)
{
    return trace.step_distance * (ceil(t / trace.step_distance - ray.phase) + ray.phase);
}

float snapToTraceDistance(float t, float stepDistance, float phase)
{
    return stepDistance * (ceil(t / stepDistance - phase) + phase);
}

#endif
