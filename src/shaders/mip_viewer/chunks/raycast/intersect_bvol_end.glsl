
// START_BLOCK_BACKWARD
#include "./modules/start_block_backward"

// MARCH_BLOCKS_BACKWARD
for (int i = MAX_BLOCKS / 2; i > 0 ; i--) 
{
    #include "./modules/update_block_backward"
    
    if (!block.empty || block.terminated) break;  
}

// UPDATE_RAY_END_IN_BLOCK
#include "./modules/update_ray_end_in_block"




