#ifndef POSITION_TO_BLOCK_COORDS
#define POSITION_TO_BLOCK_COORDS

ivec3 positionToBlockCoords(vec3 p)
{
    return ivec3(floor(p + 0.5)) / u_volume.block_size;
}

#endif
