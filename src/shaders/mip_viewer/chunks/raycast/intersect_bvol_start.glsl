
// START_BLOCK_FORWARD
#include "./modules/start_block_forward"

// MARCH_BLOCKS_FORWARD
for (int i = 0; i < MAX_BLOCKS / 2; i++) 
{
    #include "./modules/update_block_forward"

    if (!block.empty || block.terminated) break;
}

// UPDATE_RAY_START_IN_BLOCK
#include "./modules/update_ray_start_in_block"





