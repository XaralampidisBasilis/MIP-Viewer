#ifndef ADVANCE_BLOCK_COORDS
#define ADVANCE_BLOCK_COORDS

ivec3 advanceBlockCoords(ivec3 coords, ivec3 exitStep)
{
    return coords + exitStep * u_ray.signs;
}

ivec3 advanceBlockCoords(ivec3 coords, ivec3 exitStep, int stepRadius, vec3 exitPosition)
{
    ivec3 exitCoords = positionToBlockCoords(exitPosition);
    ivec3 stepCoords = coords + stepRadius * u_ray.signs;

    return exitCoords + (stepCoords - exitCoords) * exitStep;
}

#endif