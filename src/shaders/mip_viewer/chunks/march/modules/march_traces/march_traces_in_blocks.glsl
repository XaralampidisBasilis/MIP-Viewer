
#include "../march_blocks/start_block"
#include "./start_trace"
#include "./start_mip"

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    #include "../march_blocks/update_block"

    if (!(block.occupied || block.terminated)) continue;

    #include "./start_trace"
    #include "./start_mip"

    for (int i = 0; i < MAX_TRACES_IN_BLOCK; i++) 
    {
        #include "./update_trace"
        #include "./update_mip"

        if (trace.terminated || trace.distance > block.exit_distance) break;
    }   

    if (trace.terminated) break; 
}

#include "./compute_mip"
