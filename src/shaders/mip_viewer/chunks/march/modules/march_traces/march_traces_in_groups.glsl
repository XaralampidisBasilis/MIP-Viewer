
#include "../march_blocks/start_block"

#include "./start_trace"
#include "./start_mip"

for (int k = 0; k < MAX_GROUPS; k++) 
{
    #if DEBUG_ENABLED == 1

        stats.num_groups += 1;
        
    #endif

    for (int j = 0; j < MAX_BLOCKS_IN_GROUP; j++) 
    {
        #include "../march_blocks/update_block"

        if (!block.empty || block.terminated) break;
    }

    if (block.empty && !block.terminated) continue;
    
    #include "./start_trace_in_block"
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
