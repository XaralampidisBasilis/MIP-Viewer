#ifndef RAY_DISTANCE_TO_POSITION
#define RAY_DISTANCE_TO_POSITION

vec3 rayDistanceToPosition(float t)
{
    return v_ray_origin + u_ray.direction * t;
}

#endif
