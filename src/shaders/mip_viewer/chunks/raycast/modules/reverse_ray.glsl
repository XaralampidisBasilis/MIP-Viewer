#if SKIPPING_ENABLED == 1 && SKIPPING_METHOD == 1

    ray.reversed = (u_ray.reverse != 0);
    if (ray.reversed)
    {
        ray.signs = -ray.signs;
        ray.direction = -ray.direction;
        ray.inv_direction = -ray.inv_direction;
    }

#endif
