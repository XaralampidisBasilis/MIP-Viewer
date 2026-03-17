#ifndef INTERSECT_CELL_EXIT
#define INTERSECT_CELL_EXIT

float intersectCellExit(ivec3 coords, out ivec3 exitNormal)
{
    // min/max positions
    vec3 blockMin = blockCoordsToPosition(coords + 0);
    vec3 blockMax = blockCoordsToPosition(coords + 1);  

    // min/max normalized box
    vec3 tMin = (blockMin - v_ray_origin) * u_ray.inv_direction;
    vec3 tMax = (blockMax - v_ray_origin) * u_ray.inv_direction;

    vec3 tFar = max(tMin, tMax);
    float tExit = min(min(tFar.x, tFar.y), tFar.z);

    exitNormal  = ivec3(equal(tFar, vec3(tExit)));

    return tExit;

}

#endif
