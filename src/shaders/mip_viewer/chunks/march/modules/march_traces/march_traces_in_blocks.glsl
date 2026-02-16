
#include "../march_blocks/start_block"

#include "./start_mip"
#include "./start_trace"

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    #include "../march_blocks/update_block"

    if (block.terminated) break;
    if (block.shadowed) continue;
    
    #include "./start_trace"

    for (int i = 0; i < MAX_TRACES_IN_BLOCK; i++) 
    {
        #include "./update_mip"
        #include "./update_trace"

        if (trace.distance > block.exit_distance || trace.terminated) break;
    }   

    if (trace.terminated) break; 
}

#include "./compute_mip"
