#ifndef DISTANCE_TO_POSITION
#define DISTANCE_TO_POSITION

vec3 distanceToPosition(float t)
{
    return v_ray_origin + u_ray.direction * t;
}

#endif
