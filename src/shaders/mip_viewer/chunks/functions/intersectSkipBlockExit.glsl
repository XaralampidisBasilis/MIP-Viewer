#ifndef INTERSECT_SKIP_BLOCK_EXIT
#define INTERSECT_SKIP_BLOCK_EXIT


float intersectSkipBlockExit(ivec3 coords, int radius, out ivec3 exitNormal)
{
    int r = radius - 1;

    // min/max coords
    ivec3 coordsMin = coords - r;
    ivec3 coordsMax = coords + r;

    // min/max positions
    vec3 blockMin = blockCoordsToPosition(coordsMin + 0);
    vec3 blockMax = blockCoordsToPosition(coordsMax + 1);  

    // min/max normalized box
    vec3 tMin = (blockMin - v_ray_origin) * u_ray.inv_direction;
    vec3 tMax = (blockMax - v_ray_origin) * u_ray.inv_direction;

    vec3 tFar = max(tMin, tMax);
    float tExit = min(min(tFar.x, tFar.y), tFar.z);

    exitNormal  = ivec3(equal(tFar, vec3(tExit)));

    return tExit;

}

#endif
