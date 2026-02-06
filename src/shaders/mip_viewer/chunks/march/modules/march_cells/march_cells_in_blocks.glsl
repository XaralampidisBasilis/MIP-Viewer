
#include "../march_blocks/start_block"
#include "./start_cell"

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    #include "../march_blocks/update_block"

    if (block.occluded && !block.terminated) continue;

    #include "./start_cell"

    for (int i = 0; i < MAX_CELLS_IN_BLOCK; i++) 
    {
        #include "./update_cell"
        #include "./intersected_cell"

        if (cell.terminated || cell.exit_distance > block.exit_distance) break; 
    }   

    if (cell.terminated) break;
}

#include "./compute_mip"
