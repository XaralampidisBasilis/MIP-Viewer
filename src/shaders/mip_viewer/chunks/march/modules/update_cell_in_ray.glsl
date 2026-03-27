// UPDATE_CELL_IN_RAY

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
cell.terminated = cell.exit_distance > ray.end_distance - ray.eps_distance;
if (cell.terminated)
{
    cell.exit_distance = ray.end_distance;
    cell.exit_position = ray.end_position;
}

// update stats
#if DEBUG_ENABLED == 1

    stats.num_cells += 1;

#endif

