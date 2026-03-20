#ifndef INTERSECT_BLOCK
#define INTERSECT_BLOCK

struct BlockHit
{
    float entryDistance;
    float exitDistance;
    ivec3 entryStep;
    ivec3 exitStep;
};

BlockHit intersectBlock(ivec3 coords)
{
    ivec3 cMin = coords;
    ivec3 cMax = coords + 1;

    vec3 bMin = blockCoordsToPosition(cMin);
    vec3 bMax = blockCoordsToPosition(cMax);

    vec3 t0 = (bMin - v_ray_origin) * u_ray.inv_direction;
    vec3 t1 = (bMax - v_ray_origin) * u_ray.inv_direction;

    vec3 tMin = min(t0, t1);
    vec3 tMax = max(t0, t1);

    float tEntry = max(max(tMin.x, tMin.y), tMin.z);
    float tExit  = min(min(tMax.x, tMax.y), tMax.z);

    ivec3 entryStep = ivec3(
        tMin.x == tEntry ? 1 : 0,
        tMin.y == tEntry ? 1 : 0,
        tMin.z == tEntry ? 1 : 0
    );

    ivec3 exitStep = ivec3(
        tMax.x == tExit ? 1 : 0,
        tMax.y == tExit ? 1 : 0,
        tMax.z == tExit ? 1 : 0
    );

    return BlockHit(tEntry, tExit, entryStep, exitStep);
}

#endif
