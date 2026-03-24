
// START_CELL_IN_BLOCK
cell.far_distances = startCellFarDistancesInBlock(block.coords, block.entry_step, block.entry_position + ray.eps_direction);
cell.exit_distance = block.entry_distance;
cell.exit_position = block.entry_position; 
cell.exit_step = ivec3(0);
