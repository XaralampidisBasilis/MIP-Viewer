
// compute box min/max positions
cell.min_position = vec3(cell.coords) - 0.5;
cell.max_position = vec3(cell.coords) + 0.5;

// compute entry from previous exit
cell.entry_distance = cell.exit_distance;
cell.entry_position = cell.exit_position;

// compute exit from cell ray intersection 
cell.exit_distance = intersect_box_exit(cell.min_position, cell.max_position, camera.position, ray.inv_direction, cell.exit_normal);
cell.exit_position = camera.position + ray.direction * cell.exit_distance; 

// compute span distance
cell.span_distance = cell.exit_distance - cell.entry_distance;

// compute next coordinates
cell.coords += cell.exit_normal * ray.signs;

// compute termination condition
cell.terminated = cell.exit_distance > ray.end_distance; 

// update stats
#if DEBUG_ENABLED == 1
stats.num_cells += 1;
#endif
