
// compute box min/max positions in grid space
box.min_position = vec3(0.0);
box.max_position = vec3(u_volume.dimensions);

// compute rays bound distances with the box
vec2 box_min_max = intersection_box_bounds(box.min_position, box.max_position, ray.origin);

// compute current ray intersection distances with the volume box
vec2 box_entry_exit = intersect_box(box.min_position, box.max_position, ray.origin, ray.inv_direction);

// set distances to zero when we are inside
if (ray.reversed)
    box_entry_exit.y = min(box_entry_exit.y, 0.0); 
else
    box_entry_exit.x = max(box_entry_exit.x, 0.0); 

// update ray if there is an intersection 
if (box_entry_exit.x < box_entry_exit.y)
{
    // update camera box bounds
    box.min_entry_distance = box_min_max.x;
    box.max_exit_distance  = box_min_max.y;
    box.max_span_distance  = box_min_max.y - box_min_max.x;

    // update ray box distances
    box.entry_distance = box_entry_exit.x;
    box.entry_position = box_entry_exit.x * ray.direction + ray.origin;
    box.exit_distance  = box_entry_exit.y;
    box.exit_position  = box_entry_exit.y * ray.direction + ray.origin;
    box.span_distance  = box_entry_exit.y - box_entry_exit.x;
    
    // update ray distances
    ray.start_distance = box.entry_distance;
    ray.start_position = box.entry_position;
    ray.end_distance   = box.exit_distance;
    ray.end_position   = box.exit_position;
    ray.span_distance  = box.span_distance;
}