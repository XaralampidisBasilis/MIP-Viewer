
#include "../march_blocks/start_block"
#include "./start_cell"

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    #include "../march_blocks/update_block"

    if (!(block.occupied || block.terminated)) continue;

    #include "./start_cell"

    for (int i = 0; i < MAX_CELLS_IN_BLOCK; i++) 
    {
        #include "./update_cell"
        #include "./intersected_cell"

        if (cell.intersected || cell.terminated || cell.exit_distance > block.exit_distance) break; 
    }   

    if (cell.intersected || cell.terminated) break;
}

#include "./end_cell"
