
// START_BLOCK_IN_RAY
#include "../modules/start_block_in_ray"

// START_TRACE_IN_RAY  
#include "../modules/start_trace_in_ray"

// MARCH_BLOCKS
for (int k = 0; k < MAX_BLOCKS; k++) 
{
    // UPDATE_BLOCK_IN_RAY
    #include "../modules/update_block_in_ray"

    // CONTINUE_OR_BREAK_MARCH_BLOCKS
    if (block.empty)
    {
        if (!block.terminated) continue; else break;    
    }

    // START_CELL_IN_BLOCK
    #include "../modules/start_cell_in_block"

    // START_TRACE_IN_BLOCK
    #include "../modules/start_trace_in_block"
      
    // MARCH_CELLS_IN_BLOCK
    #pragma unroll
    for (int j = 0; j < MAX_CELLS_IN_BLOCK; j++)
    {
        // UPDATE_CELL_IN_BLOCK
        #include "../modules/update_cell_in_block"

        // MARCH_TRACES_IN_CELL
        #pragma unroll
        for (int i = 1; i < MAX_TRACES_IN_CELL; i++)
        {
            // UPDATE_TRACE_IN_CELL
            #include "../modules/update_trace_in_cell"

            // UPDATE_MIP_IN_TRACE
            #include "../modules/intersect_mip_in_trace"

            if (mip.intersected) break;
        }

        // BREAK_MARCH_CELLS_IN_BLOCK
        if (mip.intersected || cell.terminated) break; 
    }

    // BREAK_MARCH_BLOCKS
    if (mip.intersected || block.terminated) break;
}

// END_MIP
#include "../modules/end_mip"

