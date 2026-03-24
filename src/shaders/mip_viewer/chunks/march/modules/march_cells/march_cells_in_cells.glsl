

// start block at ray start
#include "./modules/start_cell_at_ray"

// start cubic at the ray start
#include "./modules/start_cubic_at_ray"

// start mip at the ray start
#include "./modules/start_mip_from_cubic"

// START_MARCH
for (int i = 0; i < MAX_CELLS; i++) 
{
    // update cell based on the previous one
    #include "./modules/update_cell"

    // Reconstruct the cubic polynomial inside the cell entry and exit
    #include "./modules/update_cubic_in_cell"

    // Maximize the cubic inside the cell 
    #include "./modules/maximize_cubic_in_cell"

    // Update mip based on the max cubic value
    #include "./modules/update_mip_from_cubic"

    if (cell.terminated) break; 
}

// END_MIP
#include "./modules/end_mip"



