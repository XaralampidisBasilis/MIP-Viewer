// UPDATE_CELL_IN_BLOCK

#if BLOCK_SIZE != 1

    // compute far distances
    cell.coords = advanceCellCoords(cell.coords, cell.exit_step);
    cell.far_distances = advanceCellFarDistances(cell.far_distances, cell.exit_step);

    // compute entry from previous exit
    cell.entry_distance = cell.exit_distance;
    cell.entry_position = cell.exit_position;

    // compute exit from cell ray intersection 
    cell.exit_distance = intersectCellFarDistances(cell.far_distances, cell.exit_step);
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

#else

    // copy cell from block
    cell.entry_distance = block.entry_distance;
    cell.entry_position = block.entry_position;

    cell.exit_distance = block.exit_distance;
    cell.exit_position = block.exit_position;

    cell.terminated = true;

#endif

