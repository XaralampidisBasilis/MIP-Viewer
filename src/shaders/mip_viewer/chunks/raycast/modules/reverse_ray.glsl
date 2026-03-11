#if SKIPPING_ENABLED == 1 

    ray.reversed = (ray.signs[ray.axis] < 0);
    if (ray.reversed)
    {
        ray.signs = -ray.signs;
        ray.direction = -ray.direction;
        ray.inv_direction = -ray.inv_direction;
    }

#endif
