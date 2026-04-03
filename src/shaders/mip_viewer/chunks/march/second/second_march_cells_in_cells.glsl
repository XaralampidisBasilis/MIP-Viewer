
// start block at ray start
#include "../modules/start_cell_in_ray"

// start cubic at the ray start
#include "../modules/start_cubic_in_ray"

// START_MARCH
for (int i = 0; i < MAX_CELLS; i++) 
{
    // update cell based on the previous one
    #include "../modules/update_cell_in_ray"

    // Reconstruct the cubic polynomial inside the cell entry and exit
    #include "../modules/update_cubic_in_cell"

    // Maximize the cubic inside the cell 
    #include "../modules/maximize_cubic_in_cell"

    // Update mip based on the max cubic value
    #include "../modules/intersect_mip_in_cubic"

    if (mip.intersected || cell.terminated) break; 
}

// END_MIP
#include "../modules/end_mip"



