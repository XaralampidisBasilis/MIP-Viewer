
// compute current ray intersection distances with the volume box
vec2 ray_start_end_distance = intersect_box(box.min_position, box.max_position, v_ray_origin, u_ray.inv_direction);

// update ray distances
ray.start_distance = ray_start_end_distance.x;
ray.end_distance   = ray_start_end_distance.y;
ray.span_distance  = ray.end_distance - ray.start_distance;
ray.start_position = rayDistanceToPosition(ray.start_distance); 
ray.end_position   = rayDistanceToPosition(ray.end_distance);
