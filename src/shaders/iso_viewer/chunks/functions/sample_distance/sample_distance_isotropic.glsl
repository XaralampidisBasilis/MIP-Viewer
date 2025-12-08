#ifndef SAMPLE_DISTANCE_ISOTROPIC
#define SAMPLE_DISTANCE_ISOTROPIC

// Samples the isotropic distance texture at given integer coordinates.
ivec3 sample_distance_isotropic(in ivec3 block_coords, out bool occupied)
{
    uint distance = texelFetch(u_textures.distance_map, block_coords, 0).r;
    occupied = (distance == 0u);
    return ivec3(distance);
}

#endif