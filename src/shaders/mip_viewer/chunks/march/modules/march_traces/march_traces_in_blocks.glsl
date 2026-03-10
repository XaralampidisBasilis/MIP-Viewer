
#include "../march_blocks/start_block"

#include "./set_trace"
#include "./start_mip"

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    #include "../march_blocks/update_block"

    if (block.shadowed && !block.terminated) continue;
    
    #include "./start_trace"

    for (int i = 0; i < MAX_TRACES_IN_BLOCK; i++) 
    {
        #include "./update_mip"
        #include "./update_trace"

        if (trace.terminated || trace.distance > block.exit_distance) break;
    }   

    if (trace.terminated || block.terminated) break; 
}

#include "./end_mip"
