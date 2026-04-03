
// END_RAY_IN_MIP
ray.end_distance = mip.distance + ray.eps_distance;
ray.end_position = distanceToPosition(ray.end_distance);

ray.span_distance = ray.end_distance - ray.start_distance;
