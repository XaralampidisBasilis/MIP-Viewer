
#include "./start_cubic_cell"
#include "./start_mip"

for (int i = 0; i < MAX_CELLS; i++) 
{
    #include "./update_cubic_cell"
    #include "./update_mip"

    if (cell.terminated) break;
}

#include "./end_mip"
