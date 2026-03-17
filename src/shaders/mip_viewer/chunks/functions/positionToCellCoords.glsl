#ifndef POSITION_TO_CELL_COORDS
#define POSITION_TO_CELL_COORDS

ivec3 positionToCellCoords(vec3 p)
{
    return ivec3(floor(p + 0.5));
}

#endif
