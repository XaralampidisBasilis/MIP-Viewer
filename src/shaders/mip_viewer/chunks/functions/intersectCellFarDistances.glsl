#ifndef INTERSECT_CELL_FAR_DISTANCES
#define INTERSECT_CELL_FAR_DISTANCES

float intersectCellFarDistances(vec3 tFar, out ivec3 exitStep)
{
    float tExit = min(min(tFar.x, tFar.y), tFar.z);

    exitStep = ivec3(
        tFar.x == tExit ? 1 : 0,
        tFar.y == tExit ? 1 : 0,
        tFar.z == tExit ? 1 : 0
    );

    return tExit;
}

#endif
