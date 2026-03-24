#ifndef BLOCK_COORDS_TO_POSITION
#define BLOCK_COORDS_TO_POSITION

vec3 blockCoordsToPosition(ivec3 blockCoords)
{
    return vec3(blockCoords * BLOCK_SIZE) - 0.5;
}

#endif
