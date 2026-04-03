#ifndef INTERSECT_BLOCK_EXIT
#define INTERSECT_BLOCK_EXIT

#ifndef BLOCK_COORDS_TO_MIN_POSITION
#include "./blockCoordsToMinPosition"
#endif

float intersectBlockExit(ivec3 coords, out ivec3 exitStep)
{
    ivec3 cMin = coords;
    ivec3 cMax = coords + 1;

    vec3 bMin = blockCoordsToMinPosition(cMin);
    vec3 bMax = blockCoordsToMinPosition(cMax);
    vec3 bExit = vec3(
        u_ray.sign_direction.x > 0 ? bMax.x : bMin.x,
        u_ray.sign_direction.y > 0 ? bMax.y : bMin.y,
        u_ray.sign_direction.z > 0 ? bMax.z : bMin.z
    );

    vec3 tFar = (bExit - ray.origin_position) * u_ray.inv_direction;
    float tExit = min(min(tFar.x, tFar.y), tFar.z);

    exitStep = ivec3(
        tFar.x == tExit ? 1 : 0,
        tFar.y == tExit ? 1 : 0,
        tFar.z == tExit ? 1 : 0
    );

    return tExit;
}

float intersectBlockExit(ivec3 coords, int radius, out ivec3 exitStep)
{
    ivec3 cMin = coords - radius + 1;
    ivec3 cMax = coords + radius;

    vec3 bMin = blockCoordsToMinPosition(cMin);
    vec3 bMax = blockCoordsToMinPosition(cMax);
    vec3 bExit = vec3(
        u_ray.sign_direction.x > 0 ? bMax.x : bMin.x,
        u_ray.sign_direction.y > 0 ? bMax.y : bMin.y,
        u_ray.sign_direction.z > 0 ? bMax.z : bMin.z
    );

    vec3 tFar = (bExit - ray.origin_position) * u_ray.inv_direction;
    float tExit = min(min(tFar.x, tFar.y), tFar.z);

    exitStep = ivec3(
        tFar.x == tExit ? 1 : 0,
        tFar.y == tExit ? 1 : 0,
        tFar.z == tExit ? 1 : 0
    );

    return tExit;
}

#endif
