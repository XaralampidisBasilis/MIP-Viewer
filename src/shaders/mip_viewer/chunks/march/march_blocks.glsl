
// start block at ray start
#include "./modules/start_block_in_ray"

// start mip at the ray start
#include "./modules/start_mip_in_ray"

// START_MARCH
for (int i = 0; i < MAX_BLOCKS; i++) 
{
    // Update block and get if its empty and what is the step distance we can take
    #include "./modules/update_block_in_ray"

    // CONTINUE_OR_TERMINATE_MARCH_BLOCKS
    if (block.empty)
    {
        if (!block.terminated) continue; else break;    
    }

    // Update mip with the block entry point
    #include "./modules/update_mip_in_block"

    if (block.terminated) break;
}

// END_MIP
#include "./modules/end_mip"



