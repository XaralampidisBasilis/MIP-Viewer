
// START_BLOCK_IN_RAY
#include "../modules/start_cell_in_ray"

// START_TRACE_IN_RAY  
#include "../modules/start_trace_in_ray"

// MARCH_CELLS
for (int j = 0; j < MAX_CELLS; j++) 
{
    // UPDATE_CELL_IN_RAY
    #include "../modules/update_cell_in_ray"

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

    // BREAK_MARCH_CELLS
    if (mip.intersected || cell.terminated) break; 
}

// END_MIP
#include "../modules/end_mip"



