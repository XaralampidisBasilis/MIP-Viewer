#ifdef STRUCT_RAY

ray.origin         = getRayOrigin();
ray.phase          = random(ray.origin);
ray.discarded      = false;
ray.reversed       = u_ray.reverse;
ray.direction      = u_ray.direction;
ray.step_distance  = u_ray.step_distance;
ray.eps_direction  = u_ray.direction * 0.001;
ray.eps_distance   = u_ray.step_distance * 0.001;
ray.start_position = vec3(0.0);
ray.end_position   = vec3(0.0);
ray.start_distance = 0.0;
ray.end_distance   = 0.0;
ray.span_distance  = 0.0;

#endif // STRUCT_RAY
