#ifndef POSITION_TO_BLOCK_COORDS
#define POSITION_TO_BLOCK_COORDS

ivec3 positionToBlockCoords(vec3 position)
{
    #if BLOCK_SIZE == 1
    return ivec3(floor(position + 0.5));
    #else
    return ivec3(floor(position + 0.5)) / BLOCK_SIZE;
    #endif
}

#endif
