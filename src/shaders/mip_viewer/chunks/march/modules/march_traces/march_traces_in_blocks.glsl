
#include "../march_blocks/start_block"

#include "./set_trace"
#include "./start_mip"

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    #include "../march_blocks/update_block"

    if (block.shadowed && block.exit_distance < ray.end_distance) continue;
    
    #include "./start_trace"

    for (int i = 0; i < MAX_TRACES_IN_BLOCK; i++) 
    {
        #include "./update_mip"
        #include "./update_trace"

        if (trace.distance > block.exit_distance || trace.distance > ray.end_distance) break;
    }   

    if (trace.distance > ray.end_distance) break; 
}

#include "./end_mip"
