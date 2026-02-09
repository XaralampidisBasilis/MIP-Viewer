
#include "../march_blocks/start_block"
#include "./start_cell"
#include "./start_cubic"
#include "./start_mip"

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    #include "../march_blocks/update_block"

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
}

#include "./compute_mip"

