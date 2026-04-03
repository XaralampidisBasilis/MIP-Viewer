#ifndef INTERSECT_BLOCK_ENTRY
#define INTERSECT_BLOCK_ENTRY

#ifndef BLOCK_COORDS_TO_MIN_POSITION
#include "./blockCoordsToMinPosition"
#endif

float intersectBlockEntry(ivec3 coords, out ivec3 entryStep)
{
    ivec3 cMin = coords;
    ivec3 cMax = coords + 1;

    vec3 bMin = blockCoordsToMinPosition(cMin);
    vec3 bMax = blockCoordsToMinPosition(cMax);
    vec3 bEntry = vec3(
        u_ray.sign_direction.x > 0 ? bMin.x : bMax.x,
        u_ray.sign_direction.y > 0 ? bMin.y : bMax.y,
        u_ray.sign_direction.z > 0 ? bMin.z : bMax.z
    );

    vec3 tNear = (bEntry - ray.origin_position) * u_ray.inv_direction;
    float tEntry = max(max(tNear.x, tNear.y), tNear.z);

    entryStep = ivec3(
        tNear.x == tEntry ? 1 : 0,
        tNear.y == tEntry ? 1 : 0,
        tNear.z == tEntry ? 1 : 0
    );

    return tEntry;
}

float intersectBlockEntry(ivec3 coords, int radius, out ivec3 entryStep)
{
    ivec3 cMin = coords - radius + 1;
    ivec3 cMax = coords + radius;

    vec3 bMin = blockCoordsToMinPosition(cMin);
    vec3 bMax = blockCoordsToMinPosition(cMax);
    vec3 bEntry = vec3(
        u_ray.sign_direction.x > 0 ? bMin.x : bMax.x,
        u_ray.sign_direction.y > 0 ? bMin.y : bMax.y,
        u_ray.sign_direction.z > 0 ? bMin.z : bMax.z
    );

    vec3 tNear = (bEntry - ray.origin_position) * u_ray.inv_direction;
    float tEntry = max(max(tNear.x, tNear.y), tNear.z);

    entryStep = ivec3(
        tNear.x == tEntry ? 1 : 0,
        tNear.y == tEntry ? 1 : 0,
        tNear.z == tEntry ? 1 : 0
    );

    return tEntry;
}

#endif
