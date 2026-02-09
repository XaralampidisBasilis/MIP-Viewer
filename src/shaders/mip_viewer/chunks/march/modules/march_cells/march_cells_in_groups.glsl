
#include "../march_blocks/start_block"
#include "./start_cell"
#include "./start_cubic"
#include "./start_mip"

for (int k = 0; k < MAX_GROUPS; k++) 
{
    for (int j = 0; j < MAX_BLOCKS_IN_GROUP; j++) 
    {
        #include "../march_blocks/update_block"

        if (block.terminated) break;
    }

    if (block.occluded && !block.terminated) continue;
    
    #include "./start_cell"
    #include "./start_cubic"

    for (int i = 0; i < MAX_CELLS_IN_BLOCK; i++) 
    {
        #include "./update_cell"
        #include "./update_cubic"
        #include "./update_mip"

        if (cell.terminated || cell.exit_distance > block.exit_distance) break; 
    }   

    if (cell.terminated) break;

    #if DEBUG_ENABLED == 1

        stats.num_groups += 1;

    #endif
}

#include "./compute_mip"
