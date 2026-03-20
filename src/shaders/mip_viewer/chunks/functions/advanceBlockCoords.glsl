#ifndef ADVANCE_BLOCK_COORDS
#define ADVANCE_BLOCK_COORDS

ivec3 advanceBlockCoords(ivec3 coords, ivec3 exitStep)
{
    return coords + exitStep * u_ray.signs;
}

ivec3 advanceBlockCoords(ivec3 coords, vec3 exitPosition, int stepRadius, ivec3 exitStep)
{
    ivec3 exitCoords = positionToBlockCoords(exitPosition);
    ivec3 stepCoords = coords + stepRadius * u_ray.signs;

    return exitCoords + (stepCoords - exitCoords) * exitStep;
}

#endif