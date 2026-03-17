#ifndef CELL_COORDS_TO_POSITION
#define CELL_COORDS_TO_POSITION

vec3 cellCoordsToPosition(ivec3 c)
{
    return vec3(c) - 0.5;
}

#endif
