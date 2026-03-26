
#if BLOCK_SIZE != 1

// START_CELL_IN_BLOCK
cell.coords = startCellCoordsInBlock(block.coords, block.entry_step, block.entry_position + ray.eps_direction);
cell.far_distances = cellCoordsToFarDistances(cell.coords);

cell.exit_distance = block.entry_distance;
cell.exit_position = block.entry_position; 
cell.exit_step = ivec3(0);

#endif
