#ifndef CELL_COORDS_TO_MIN_POSITION
#define CELL_COORDS_TO_MIN_POSITION

vec3 cellCoordsToMinPosition(ivec3 coords)
{
    return vec3(coords) - 0.5;
}

#endif
