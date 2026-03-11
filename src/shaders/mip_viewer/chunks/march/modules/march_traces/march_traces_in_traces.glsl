
#include "./set_trace"
#include "./start_mip"

for (int i = 0; i < MAX_TRACES; i++) 
{
    #include "./update_trace_position"

    if (trace.terminated) break;

    #include "./update_trace_value"
    #include "./update_mip"
}

#include "./end_mip"
