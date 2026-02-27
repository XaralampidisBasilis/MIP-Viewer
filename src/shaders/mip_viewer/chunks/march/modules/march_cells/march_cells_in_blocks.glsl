
#include "../march_blocks/start_block"

#include "./start_cubic_cell"
#include "./start_mip"

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    #include "../march_blocks/update_block"

    if (block.shadowed && block.exit_distance < ray.end_distance) continue;

    #include "./start_cubic_cell"

    for (int i = 0; i < MAX_CELLS_IN_BLOCK; i++) 
    {
        #include "./update_cubic_cell"
        #include "./update_mip"

        if (cell.exit_distance > block.exit_distance || cell.exit_distance > ray.end_distance) break; 
    }   

    if (cell.exit_distance > ray.end_distance) break;
}

#include "./compute_mip"

