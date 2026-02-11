
#include "../march_blocks/start_block"
#include "./start_trace"
#include "./start_mip"

for (int k = 0; k < MAX_GROUPS; k++) 
{
    for (int j = 0; j < MAX_BLOCKS_IN_GROUP; j++) 
    {
        #include "../march_blocks/update_block"

        if (!block.shadowed || block.terminated) break;
    }

    if (block.shadowed && !block.terminated) continue;

    #include "./start_trace"
    #include "./update_mip"

    for (int i = 0; i < MAX_TRACES_IN_BLOCK; i++) 
    {
        #include "./update_trace"
        #include "./update_mip"

        if (trace.terminated || trace.distance > block.exit_distance) break;
    }   

    if (trace.terminated) break; 

    #if DEBUG_ENABLED == 1

        stats.num_groups += 1;
        
    #endif
}

#include "./compute_mip"
