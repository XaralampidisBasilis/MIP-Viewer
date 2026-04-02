
#ifndef HALF_BLOCKS
#define HALF_BLOCKS MAX_BLOCKS / 2
#endif

// START_BLOCK_FORWARD
#include "./modules/start_block_forward"

// MARCH_BLOCKS_FORWARD
for (int i = 0; i < HALF_BLOCKS; i++) 
{
    #include "./modules/update_block_forward"

    if (!block.empty || block.terminated) break;
}

// UPDATE_RAY_START_IN_BLOCK
#include "./modules/update_ray_start_in_block"

// START_BLOCK_BACKWARD
#include "./modules/start_block_backward"

// MARCH_BLOCKS_BACKWARD
for (int i = 0; i < HALF_BLOCKS; i++) 
{
    #include "./modules/update_block_backward"
    
    if (!block.empty || block.terminated) break;  
}

// UPDATE_RAY_END_IN_BLOCK
#include "./modules/update_ray_end_in_block"

// RAY DISCARD CONDITION
if (ray.start_distance > ray.end_distance)
{
    ray.discarded = true;
}




