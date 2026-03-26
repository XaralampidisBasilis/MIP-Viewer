
// START_CELL_IN_RAY
cell.coords = positionToCellCoords(ray.start_position);
cell.far_distances = cellCoordsToFarDistances(cell.coords);

cell.exit_distance = ray.start_distance;
cell.exit_position = ray.start_position; 
cell.exit_step = ivec3(0);
