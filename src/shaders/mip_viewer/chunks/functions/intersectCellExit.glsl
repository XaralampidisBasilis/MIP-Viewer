#ifndef INTERSECT_CELL_EXIT
#define INTERSECT_CELL_EXIT

float intersectCellExit(ivec3 coords, out ivec3 exitStep)
{
    ivec3 cMin = coords;
    ivec3 cMax = coords + 1;

    vec3 bMin = cellCoordsToPosition(cMin);
    vec3 bMax = cellCoordsToPosition(cMax);
    
    vec3 bExit = vec3(
        u_ray.signs.x > 0 ? bMax.x : bMin.x,
        u_ray.signs.y > 0 ? bMax.y : bMin.y,
        u_ray.signs.z > 0 ? bMax.z : bMin.z
    );

    vec3 tFar = (bExit - v_ray_origin) * u_ray.inv_direction;
    float tExit = min(min(tFar.x, tFar.y), tFar.z);

    exitStep = ivec3(
        tFar.x == tExit ? 1 : 0,
        tFar.y == tExit ? 1 : 0,
        tFar.z == tExit ? 1 : 0
    );

    return tExit;
}

#endif
