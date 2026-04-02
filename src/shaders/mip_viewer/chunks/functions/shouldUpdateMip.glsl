#ifndef SHOULD_UPDATE_MIP
#define SHOULD_UPDATE_MIP

bool shouldUpdateMip(float newValue, float mipValue)
{
    float minValue = min(u_ray.min_value, u_ray.max_value);
    float maxValue = max(u_ray.min_value, u_ray.max_value);

    if (newValue < minValue || newValue > maxValue)
    {
        return false;
    }

    if (newValue > mipValue)
    {
        return true;
    }

    return false;
}

#endif
