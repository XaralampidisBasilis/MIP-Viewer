
#include "../march_blocks/start_block"

#include "./set_trace"
#include "./start_mip"

for (int k = 0; k < MAX_GROUPS; k++) 
{
    for (int j = 0; j < MAX_BLOCKS_IN_GROUP; j++) 
    {
        #include "../march_blocks/update_block"

        if (!block.shadowed || block.exit_distance > ray.end_distance) break;

    }

    if (block.shadowed && block.exit_distance < ray.end_distance) continue;

    #include "./start_trace"

    for (int i = 0; i < MAX_TRACES_IN_BLOCK; i++) 
    {
        #include "./update_mip"
        #include "./update_trace"

        if (trace.distance > block.exit_distance || trace.distance > ray.end_distance) break;
    }    

    if (trace.distance > ray.end_distance) break; 

    #if DEBUG_ENABLED == 1

        stats.num_groups += 1;
        
    #endif
}

#include "./end_mip"
