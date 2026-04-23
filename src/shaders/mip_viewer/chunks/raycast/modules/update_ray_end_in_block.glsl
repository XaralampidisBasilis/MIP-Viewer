
// UPDATE_RAY_END_IN_BLOCK
ray.end_distance = block.exit_distance + ray.eps_distance;
ray.end_position = distanceToPosition(ray.end_distance);
ray.span_distance = ray.end_distance - ray.start_distance;

