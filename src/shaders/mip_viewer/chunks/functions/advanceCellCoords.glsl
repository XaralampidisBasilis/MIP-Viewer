#ifndef ADVANCE_CELL_COORDS
#define ADVANCE_CELL_COORDS

ivec3 advanceCellCoords(ivec3 coords, ivec3 exitStep)
{
    return coords + exitStep * u_ray.signs;
}

ivec3 advanceCellCoords(ivec3 coords, vec3 exitPosition, int stepRadius, ivec3 exitStep)
{
    ivec3 exitCoords = positionToCellCoords(exitPosition);
    ivec3 stepCoords = coords + stepRadius * u_ray.signs;

    return exitCoords + (stepCoords - exitCoords) * exitStep;
}

/*

// Dynamic indexing versions using axis

ivec3 advanceCellCoords(ivec3 coords, int exitAxis)
{
    coords[exitAxis] += u_ray.signs[exitAxis];
    return coords;
}

ivec3 advanceCellCoords(ivec3 coords, vec3 exitPosition, int stepRadius, int exitAxis)
{
    vec3 rayNudge = u_ray.direction * eps;
    ivec3 outCoords = positionToCellCoords(exitPosition + rayNudge);
    outCoords[exitAxis] = coords[exitAxis] + stepRadius * u_ray.signs[exitAxis];

    return outCoords;
}

// Avoid dynamic indexing versions when using axis

ivec3 advanceCellCoords(ivec3 coords, int exitAxis)
{
         if (exitAxis == 0) coords.x += u_ray.signs.x;
    else if (exitAxis == 1) coords.y += u_ray.signs.y;
    else                    coords.z += u_ray.signs.z;
    return coords;
}

ivec3 advanceCellCoords(ivec3 coords, vec3 exitPosition, int stepRadius, int exitAxis)
{
    ivec3 outCoords = positionToCellCoords(exitPosition);

         if (exitAxis == 0) outCoords.x = coords.x + stepRadius * u_ray.signs.x;
    else if (exitAxis == 1) outCoords.y = coords.y + stepRadius * u_ray.signs.y;
    else                    outCoords.z = coords.z + stepRadius * u_ray.signs.z;

    return outCoords;
}

*/

#endif