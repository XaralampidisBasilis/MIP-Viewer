
#include "../march_blocks/start_block"
#include "./start_trace"
#include "./start_mip"

for (int j = 0; j < int(u_debug.max_blocks); j++) 
{
    #include "../march_blocks/update_block"

    if (block.occluded && !block.terminated) continue;

    #include "./start_trace"

    for (int i = 0; i < int(u_debug.max_cells); i++) 
    {
        #include "./update_mip"
        #include "./update_trace"

        if (trace.terminated || trace.distance > block.exit_distance) break;
    }   

    if (trace.terminated) break; 
}

#include "./compute_mip"
