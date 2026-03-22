#ifndef POSITION_TO_CELL_COORDS
#define POSITION_TO_CELL_COORDS

ivec3 positionToCellCoords(vec3 position)
{
    return ivec3(floor(position + 0.5));
}

#endif
