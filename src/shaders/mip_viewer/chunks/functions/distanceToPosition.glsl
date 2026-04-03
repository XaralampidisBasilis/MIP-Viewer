#ifndef DISTANCE_TO_POSITION
#define DISTANCE_TO_POSITION

vec3 distanceToPosition(float t)
{
    return ray.origin_position + u_ray.direction * t;
}

#endif
