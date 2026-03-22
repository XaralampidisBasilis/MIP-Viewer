#ifndef CELL_COORDS_TO_POSITION
#define CELL_COORDS_TO_POSITION

vec3 cellCoordsToPosition(ivec3 cellCoords)
{
    return vec3(cellCoords) - 0.5;
}

#endif
