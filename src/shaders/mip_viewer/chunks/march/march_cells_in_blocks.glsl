
// start block at ray start
#include "./modules/start_block_in_ray"

// start cubic at the ray start
#include "./modules/start_cubic_in_ray"

// start mip at the ray start
#include "./modules/start_mip_in_cubic"

// START_MARCH
for (int j = 0; j < u_debug.max_blocks; j++) 
{
    // Update block and get if its empty and what is the step distance we can take
    #include "./modules/update_block_in_ray"

    // CONTINUE_OR_TERMINATE_MARCH_BLOCKS
    if (block.empty)
    {
        if (!block.terminated) continue; else break;    
    }

    // Start cell at the block entry
    #include "./modules/start_cell_in_block"

    // Start cubic at the block entry and reuse sample if previous was not empty
    #include "./modules/start_cubic_in_block"
      
    // Start cell march inside the current non empty block until we escape
    #pragma unroll
    for (int i = 0; i < MAX_CELLS_IN_BLOCK; i++)
    {
        // update cell based on the previous one
        #include "./modules/update_cell_in_block"

        // Reconstruct the cubic polynomial inside the cell entry and exit
        #include "./modules/update_cubic_in_cell"

        // Maximize the cubic inside the cell 
        #include "./modules/maximize_cubic_in_cell"

        // Update mip based on the max cubic value
        #include "./modules/update_mip_in_cubic"

        if (cell.terminated) break; 
    }

    if (block.terminated) break;
}

// END_MIP
#include "./modules/end_mip"



