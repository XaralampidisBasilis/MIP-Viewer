
#include "../march_blocks/start_block"

#include "./set_trace"
#include "./start_mip"

for (int j = 0; j < u_debug.max_blocks; j++) 
{
    #include "../march_blocks/update_block"

    if (block.shadowed && !block.terminated) continue;
    
    #include "./start_trace"
    #include "./update_mip"

    for (int i = 0; i < MAX_TRACES_IN_BLOCK; i++) 
    {
        #include "./update_trace_position"

        if (trace.distance > block.exit_distance || trace.terminated) break;

        #include "./update_trace_value"
        #include "./update_mip"
    }   

    if (trace.terminated) break; 
}

#include "./end_mip"
