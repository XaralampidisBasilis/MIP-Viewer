#ifndef BLOCK_COORDS_TO_MIN_POSITION
#define BLOCK_COORDS_TO_MIN_POSITION

vec3 blockCoordsToMinPosition(ivec3 coords)
{
    #if BLOCK_SIZE == 1
    return vec3(coords) - 0.5;
    #else
    return vec3(coords * BLOCK_SIZE) - 0.5;
    #endif
}

#endif
