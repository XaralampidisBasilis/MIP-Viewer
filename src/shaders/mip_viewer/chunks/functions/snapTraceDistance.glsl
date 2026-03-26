#ifndef SNAP_TRACE_DISTANCE
#define SNAP_TRACE_DISTANCE

float snapTraceDistanceFloor(float t)
{
    return trace.step_distance * (floor(t / trace.step_distance - ray.phase) + ray.phase);
}

float snapTraceDistanceCeil(float t)
{
    return trace.step_distance * (ceil(t / trace.step_distance - ray.phase) + ray.phase);
}

float snapTraceDistanceFloor(float t, float step_distance, float phase)
{
    return step_distance * (floor(t / step_distance - phase) + phase);
}

float snapTraceDistanceCeil(float t, float step_distance, float phase)
{
    return step_distance * (ceil(t / step_distance - phase) + phase);
}

#endif
