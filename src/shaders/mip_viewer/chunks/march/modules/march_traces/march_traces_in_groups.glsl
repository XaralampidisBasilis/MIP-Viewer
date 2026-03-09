
#include "../march_blocks/start_block"

#include "./start_mip"
#include "./start_trace"

for (int k = 0; k < MAX_GROUPS; k++) 
{
    for (int j = 0; j < MAX_BLOCKS_IN_GROUP; j++) 
    {
        #include "../march_blocks/update_block"

        if (!block.shadowed || block.terminated) break;
    }

    if (block.shadowed && !block.terminated) continue;

    #include "./start_trace"

    for (int i = 0; i < MAX_TRACES_IN_BLOCK; i++) 
    {
        #include "./update_mip"
        #include "./update_trace"

        if (trace.distance > block.exit_distance || trace.terminated) break;
    }   

    if (trace.terminated) break; 

    #if DEBUG_ENABLED == 1

        stats.num_groups += 1;
        
    #endif
}

#include "./end_mip"
