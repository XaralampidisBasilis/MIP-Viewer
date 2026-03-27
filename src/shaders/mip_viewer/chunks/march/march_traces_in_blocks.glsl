
// START_BLOCK_IN_RAY
#include "./modules/start_block_in_ray"

// START_TRACE_IN_RAY
#include "./modules/start_trace_in_ray"

// START_MIP_IN_TRACE
#include "./modules/start_mip_in_trace"

// MARCH_BLOCKS
for (int j = 0; j < MAX_BLOCKS; j++) 
{
    // UPDATE_BLOCK_IN_RAY
    #include "./modules/update_block_in_ray"

    // CONTINUE_OR_TERMINATE_MARCH_BLOCKS
    if (block.empty)
    {
        if (!block.terminated) continue; else break;    
    }

    // START_TRACE_IN_BLOCK
    #include "./modules/start_trace_position_in_block"

    // MARCH_TRACES_IN_BLOCK
    #pragma unroll
    for (int i = 0; i < MAX_TRACES_IN_BLOCK; i++)
    {
        // UPDATE_TRACE_POSITION_IN_BLOCK
        #include "./modules/update_trace_position_in_block"

        // TERMINATE_MARCH_TRACES_IN_BLOCK
        if (trace.terminated) break; 

        // UPDATE_TRACE_VALUE
        #include "./modules/update_trace_value"

        // UPDATE_MIP_IN_TRACE
        #include "./modules/update_mip_in_trace"
    }

    // TERMINATE_MARCH_BLOCKS
    if (block.terminated) break;
}

// END_MIP
#include "./modules/end_mip"



