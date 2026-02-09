
#include "./start_cell"
#include "./start_cubic"
#include "./start_mip"

for (int i = 0; i < MAX_CELLS; i++) 
{
    #include "./update_cell"
    #include "./update_cubic"
    #include "./update_mip"

    if (cell.terminated) break;
}

#include "./compute_mip"
