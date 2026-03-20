#ifndef INTERSECT_BLOCK
#define INTERSECT_BLOCK

struct BlockHit
{
    float entryDistance;
    float exitDistance;
    int   entryAxis;
    int   exitAxis;
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

    int entryAxis = 0;
    if (tMin.y > tMin.x) entryAxis = 1;
    if (tMin.z > tMin[entryAxis]) entryAxis = 2;

    int exitAxis = 0;
    if (tMax.y < tMax.x) exitAxis = 1;
    if (tMax.z < tMax[exitAxis]) exitAxis = 2;

    float entryDistance = tMin[entryAxis];
    float exitDistance  = tMax[exitAxis];

    return BlockHit(entryDistance, exitDistance, entryAxis, exitAxis);
}

#endif