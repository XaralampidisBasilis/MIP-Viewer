
// UPDATE_RAY_START_IN_BLOCK
ray.start_distance = block.entry_distance - ray.eps_distance;
ray.start_position = distanceToPosition(ray.start_distance);
ray.span_distance = ray.end_distance - ray.start_distance;
