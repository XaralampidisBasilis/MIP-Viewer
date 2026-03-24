#ifndef POSITION_TO_BLOCK_COORDS
#define POSITION_TO_BLOCK_COORDS

ivec3 positionToBlockCoords(vec3 position)
{
    return ivec3(floor(position + 0.5)) / BLOCK_SIZE;
}

#endif
