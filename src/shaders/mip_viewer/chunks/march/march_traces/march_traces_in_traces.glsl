
// start trace at ray start
#include "./modules/start_trace_in_ray"

// start mip at the ray start
#include "./modules/start_mip_in_trace"

// START_MARCH
for (int i = 0; i < MAX_TRACES; i++) 
{
    // update cell based on the previous one
    #include "./modules/update_trace_in_ray"

    // Update mip based on the max cubic value
    #include "./modules/update_mip_in_trace"

    if (trace.terminated) break; 
}

// END_MIP
#include "./modules/end_mip"



