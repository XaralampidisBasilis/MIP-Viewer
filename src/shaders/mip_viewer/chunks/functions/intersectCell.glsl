#ifndef INTERSECT_CELL
#define INTERSECT_CELL

struct CellHit
{
    float entryDistance;
    float exitDistance;
    ivec3 entryStep;
    ivec3 exitStep;
};

CellHit intersectCell(ivec3 coords)
{
    ivec3 cMin = coords;
    ivec3 cMax = coords + 1;

    vec3 bMin = cellCoordsToPosition(cMin);
    vec3 bMax = cellCoordsToPosition(cMax);

    vec3 t0 = (bMin - v_ray_origin) * u_ray.inv_direction;
    vec3 t1 = (bMax - v_ray_origin) * u_ray.inv_direction;

    vec3 tMin = min(t0, t1);
    vec3 tMax = max(t0, t1);

    float tEntry = max(max(tMin.x, tMin.y), tMin.z);
    float tExit  = min(min(tMax.x, tMax.y), tMax.z);

    ivec3 entryStep = ivec3(equal(tMin, vec3(tEntry)));
    ivec3 exitStep = ivec3(equal(tMax, vec3(tExit)));

    return CellHit(tEntry, tExit, entryStep, exitStep);
}

// The function builds a box starting from the current cell, 
// then stretches that box outward on the side the ray is moving toward. 
// After that, it performs a standard ray-box intersection and 
// returns the entry time, exit time, and which faces were crossed.
CellHit intersectCell(ivec3 coords, int radius)
{
    int extent = radius - 1;

    ivec3 extMin = max(-u_ray.signs, 0) * extent;
    ivec3 extMax = max( u_ray.signs, 0) * extent;

    ivec3 cMin = coords - extMin;
    ivec3 cMax = coords + extMax + 1;

    vec3 bMin = cellCoordsToPosition(cMin);
    vec3 bMax = cellCoordsToPosition(cMax);

    vec3 t0 = (bMin - v_ray_origin) * u_ray.inv_direction;
    vec3 t1 = (bMax - v_ray_origin) * u_ray.inv_direction;

    vec3 tMin = min(t0, t1);
    vec3 tMax = max(t0, t1);

    float tEntry = max(max(tMin.x, tMin.y), tMin.z);
    float tExit  = min(min(tMax.x, tMax.y), tMax.z);

    ivec3 entryStep = ivec3(equal(tMin, vec3(tEntry)));
    ivec3 exitStep = ivec3(equal(tMax, vec3(tExit)));

    return CellHit(tEntry, tExit, entryStep, exitStep);
}

#endif
