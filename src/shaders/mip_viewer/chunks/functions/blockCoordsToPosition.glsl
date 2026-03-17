#ifndef BLOCK_COORDS_TO_POSITION
#define BLOCK_COORDS_TO_POSITION

vec3 blockCoordsToPosition(ivec3 c)
{
    return vec3(c * u_volume.block_size) - 0.5;
}

#endif
