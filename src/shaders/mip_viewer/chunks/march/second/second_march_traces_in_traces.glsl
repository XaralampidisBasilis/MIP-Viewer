
// START_TRACE_IN_RAY
#include "../modules/start_trace_in_ray"

// MARCH_TRACES
for (int i = 0; i < MAX_TRACES; i++) 
{
    // UPDATE_TRACE_POSITION_IN_RAY
    #include "../modules/update_trace_position_in_ray"

    // TERMINATE_MARCH_TRACES
    if (trace.terminated) break; 

    // UPDATE_TRACE_VALUE
    #include "../modules/update_trace_value"

    // UPDATE_MIP_IN_TRACE
    #include "../modules/intersect_mip_in_trace"

    if (mip.intersected) break;
}

// END_MIP
#include "../modules/end_mip"



