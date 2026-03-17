
// compute current ray intersection distances with the volume box
vec2 ray_start_end_distance = intersect_box(box.min_position, box.max_position, ray.origin, ray.inv_direction);

// update ray distances
ray.start_distance = ray_start_end_distance.x;
ray.end_distance   = ray_start_end_distance.y;
ray.span_distance  = ray.end_distance - ray.start_distance;
ray.start_position = rayDistanceToPosition(ray.start_distance); 
ray.end_position   = rayDistanceToPosition(ray.end_distance);

// update ray box distances
box.entry_distance = ray.start_distance;
box.exit_distance  = ray.end_distance;
box.span_distance  = ray.span_distance;
box.entry_position = ray.start_position;
box.exit_position  = ray.end_position;

