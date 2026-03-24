
// UPDATE_CELL

// compute next coordinates
cell.coords = advanceCellCoords(cell.coords, cell.exit_step);

// compute entry from previous exit
cell.entry_distance = cell.exit_distance;
cell.entry_position = cell.exit_position;

// compute exit from cell ray intersection 
cell.exit_distance = intersectCellExit(cell.coords, cell.exit_step);
cell.exit_position = distanceToPosition(cell.exit_distance);

// compute span distance
cell.span_distance = cell.exit_distance - cell.entry_distance;

// compute termination condition
cell.terminated = 
    cell.exit_distance > block.exit_distance - ray.eps_distance || 
    cell.exit_distance > ray.end_distance;

// update stats
#if DEBUG_ENABLED == 1

    stats.num_cells += 1;

#endif