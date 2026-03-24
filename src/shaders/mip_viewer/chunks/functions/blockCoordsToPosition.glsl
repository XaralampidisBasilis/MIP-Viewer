#ifndef BLOCK_COORDS_TO_POSITION
#define BLOCK_COORDS_TO_POSITION

vec3 blockCoordsToPosition(ivec3 blockCoords)
{
    #if BLOCK_SIZE == 1
    return vec3(blockCoords) - 0.5;
    #else
    return vec3(blockCoords * BLOCK_SIZE) - 0.5;
    #endif
}

#endif
