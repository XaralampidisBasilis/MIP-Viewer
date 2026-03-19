#ifndef INTERSECT_SKIP_CELL_EXIT
#define INTERSECT_SKIP_CELL_EXIT

float intersectSkipCellExit(ivec3 coords, int radius, out ivec3 exitNormal)
{
    ivec3 cMin = coords + ivec3(1 - radius);
    ivec3 cMax = coords + ivec3(radius);

    vec3 boxMin = cellCoordsToPosition(cMin);
    vec3 boxMax = cellCoordsToPosition(cMax);

    vec3 boxExit = vec3(
        u_ray.signs.x > 0 ? boxMax.x : boxMin.x,
        u_ray.signs.y > 0 ? boxMax.y : boxMin.y,
        u_ray.signs.z > 0 ? boxMax.z : boxMin.z
    );

    vec3 tFar = (boxExit - v_ray_origin) * u_ray.inv_direction;
    float tExit = min(min(tFar.x, tFar.y), tFar.z);

    exitNormal = ivec3(
        tFar.x == tExit ? 1 : 0,
        tFar.y == tExit ? 1 : 0,
        tFar.z == tExit ? 1 : 0
    );

    return tExit;
}

// float intersectSkipCellExit(ivec3 coords, int radius, out ivec3 exitNormal)
// {
//     int r = radius - 1;

//     // min/max coords
//     ivec3 cMin = coords - r;
//     ivec3 cMax = coords + r;

//     // min/max positions
//     vec3 boxMin = cellCoordsToPosition(cMin);
//     vec3 boxMax = cellCoordsToPosition(cMax + 1);  

//     // min/max normalized box
//     vec3 tMin = (boxMin - v_ray_origin) * u_ray.inv_direction;
//     vec3 tMax = (boxMax - v_ray_origin) * u_ray.inv_direction;

//     vec3 tFar = max(tMin, tMax);
//     float tExit = min(min(tFar.x, tFar.y), tFar.z);

//     exitNormal  = ivec3(equal(tFar, vec3(tExit)));

//     return tExit;
// }

#endif
