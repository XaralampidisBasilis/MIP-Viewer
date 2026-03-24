#ifndef ADVANCE_CELL_COORDS
#define ADVANCE_CELL_COORDS

ivec3 advanceCellCoords(ivec3 coords, ivec3 exitStep)
{
    return coords + exitStep * u_ray.sign_direction;
}

#endif