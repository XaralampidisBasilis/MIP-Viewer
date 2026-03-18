#ifndef INTERSECT_CELL_EXIT
#define INTERSECT_CELL_EXIT

float intersectCellExit(ivec3 coords, out ivec3 exitNormal)
{
    ivec3 cMin = coords;
    ivec3 cMax = coords + ivec3(1);

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

// float intersectCellExit(ivec3 coords, out ivec3 exitNormal)
// {
//     // min/max positions
//     vec3 blockMin = blockCoordsToPosition(coords + 0);
//     vec3 blockMax = blockCoordsToPosition(coords + 1);  

//     // min/max normalized box
//     vec3 tMin = (blockMin - v_ray_origin) * u_ray.inv_direction;
//     vec3 tMax = (blockMax - v_ray_origin) * u_ray.inv_direction;

//     vec3 tFar = max(tMin, tMax);
//     float tExit = min(min(tFar.x, tFar.y), tFar.z);

//     exitNormal  = ivec3(equal(tFar, vec3(tExit)));

//     return tExit;
// }

#endif
