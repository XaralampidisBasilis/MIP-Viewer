
#include "../march_blocks/start_block"

#include "./start_trace"
#include "./start_mip"

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    #include "../march_blocks/update_block"

    if (block.shadowed && block.exit_distance < ray.end_distance) continue;
    
    #include "./start_trace"
    #include "./update_mip"

    for (int i = 0; i < MAX_TRACES_IN_BLOCK; i++) 
    {
        #include "./update_trace"
        #include "./update_mip"

        if (trace.distance > block.exit_distance) break;
    }   

    if (trace.distance > ray.end_distance) break; 
}

#include "./compute_mip"
