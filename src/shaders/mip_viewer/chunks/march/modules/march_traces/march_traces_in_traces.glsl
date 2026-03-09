
#include "./set_trace"
#include "./start_mip"

for (int i = 0; i < MAX_TRACES; i++) 
{
    #include "./update_trace"
    #include "./update_mip"

    if (trace.terminated) break;
}

#include "./end_mip"
