#ifndef INTERSECT_CELL_EXIT
#define INTERSECT_CELL_EXIT

#ifndef CELL_COORDS_TO_MIN_POSITION
#include "./cellCoordsToMinPosition"
#endif

float intersectCellExit(ivec3 coords, out ivec3 exitStep)
{
    vec3 cMin = cellCoordsToMinPosition(coords);
    vec3 cMax = cMin + vec3(1.0);
    vec3 cFar = vec3(
        u_ray.sign_direction.x > 0 ? cMax.x : cMin.x,
        u_ray.sign_direction.y > 0 ? cMax.y : cMin.y,
        u_ray.sign_direction.z > 0 ? cMax.z : cMin.z
    );

    vec3 tFar = (cFar - v_ray_origin) * u_ray.inv_direction;
    float tExit = min(min(tFar.x, tFar.y), tFar.z);

    exitStep = ivec3(
        tFar.x == tExit ? 1 : 0,
        tFar.y == tExit ? 1 : 0,
        tFar.z == tExit ? 1 : 0
    );

    return tExit;
}

#endif
