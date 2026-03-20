#ifndef INTERSECT_CELL_EXIT
#define INTERSECT_CELL_EXIT

float intersectCellExit(ivec3 coords, out ivec3 exitStep)
{
    ivec3 cMin = coords;
    ivec3 cMax = coords + 1;

    vec3 boxMin = cellCoordsToPosition(cMin);
    vec3 boxMax = cellCoordsToPosition(cMax);
    
    vec3 boxExit = vec3(
        u_ray.signs.x > 0 ? boxMax.x : boxMin.x,
        u_ray.signs.y > 0 ? boxMax.y : boxMin.y,
        u_ray.signs.z > 0 ? boxMax.z : boxMin.z
    );

    vec3 tFar = (boxExit - v_ray_origin) * u_ray.inv_direction;
    float tExit = min(min(tFar.x, tFar.y), tFar.z);

    exitStep = ivec3(
        tFar.x == tExit ? 1 : 0,
        tFar.y == tExit ? 1 : 0,
        tFar.z == tExit ? 1 : 0
    );

    return tExit;
}

float intersectCellExit(ivec3 coords, int radius, out ivec3 exitStep)
{
    ivec3 cMin = coords - radius + 1;
    ivec3 cMax = coords + radius;

    vec3 boxMin = cellCoordsToPosition(cMin);
    vec3 boxMax = cellCoordsToPosition(cMax);

    vec3 boxExit = vec3(
        u_ray.signs.x > 0 ? boxMax.x : boxMin.x,
        u_ray.signs.y > 0 ? boxMax.y : boxMin.y,
        u_ray.signs.z > 0 ? boxMax.z : boxMin.z
    );

    vec3 tFar = (boxExit - v_ray_origin) * u_ray.inv_direction;
    float tExit = min(min(tFar.x, tFar.y), tFar.z);

    exitStep = ivec3(
        tFar.x == tExit ? 1 : 0,
        tFar.y == tExit ? 1 : 0,
        tFar.z == tExit ? 1 : 0
    );

    return tExit;
}

#endif
