
#include "../march_blocks/start_block"
#include "./start_trace"

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    #include "../march_blocks/update_block"

    if (!(block.occupied || block.terminated)) continue;

    #include "./start_trace"

    for (int i = 0; i < MAX_TRACES_IN_BLOCK; i++) 
    {
        #include "./update_trace"
        #include "./intersected_trace"

        if (trace.intersected || trace.terminated || trace.distance > block.exit_distance) break;
    }   

    if (trace.intersected || trace.terminated) break; 
}

#include "./end_trace"
