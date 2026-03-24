
// UPDATE_CELL

// compute next coordinates
cell.coords = block.coords;

// compute entry from previous exit
cell.entry_distance = block.entry_distance;
cell.entry_position = block.entry_position;

// compute exit from cell ray intersection 
cell.exit_distance = block.exit_distance;
cell.exit_position = block.exit_position;
cell.exit_step = block.exit_step;

// compute span distance
cell.span_distance = block.span_distance;

// compute termination condition
cell.terminated = block.terminated;

// update stats
#if DEBUG_ENABLED == 1

    stats.num_cells += 1;

#endif