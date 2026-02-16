
#include "./start_mip"
#include "./start_trace"

for (int i = 0; i < MAX_TRACES; i++) 
{
    #include "./update_mip"
    #include "./update_trace"

    if (trace.terminated) break;
}

#include "./compute_mip"
