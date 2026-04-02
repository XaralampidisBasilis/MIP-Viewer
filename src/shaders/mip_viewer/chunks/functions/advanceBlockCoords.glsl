#ifndef ADVANCE_BLOCK_COORDS
#define ADVANCE_BLOCK_COORDS

#ifndef POSITION_TO_BLOCK_COORDS
#include "./positionToBlockCoords"
#endif

ivec3 advanceBlockCoords(ivec3 coords, ivec3 stepVector)
{
    ivec3 stepNormal = stepVector * u_ray.sign_direction;
    return coords + stepNormal;
}

ivec3 advanceBlockCoords(ivec3 coords, ivec3 stepVector, int stepRadius, vec3 position)
{
    ivec3 stepNormal = stepVector * u_ray.sign_direction;
    if (stepRadius == 1) return coords + stepNormal;

    ivec3 exitCoords = positionToBlockCoords(position);
    ivec3 stepCoords = coords + stepRadius * stepNormal;

    return exitCoords + (stepCoords - exitCoords) * stepVector;
}

#endif