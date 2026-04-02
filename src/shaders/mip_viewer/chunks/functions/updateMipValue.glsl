#ifndef UPDATE_MIP_VALUE
#define UPDATE_MIP_VALUE

bool updateMipValue(float newValue, float mipValue)
{
    if (newValue < u_ray.min_value || newValue > u_ray.max_value)
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