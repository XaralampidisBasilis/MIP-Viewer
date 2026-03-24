
// START_CELL_AT_BLOCK
cell.coords = startCellCoordsInBlock(block.coords, block.entry_step, block.entry_position + ray.eps_direction);
cell.exit_distance = block.entry_distance;
cell.exit_position = block.entry_position; 
cell.exit_step = ivec3(0);
