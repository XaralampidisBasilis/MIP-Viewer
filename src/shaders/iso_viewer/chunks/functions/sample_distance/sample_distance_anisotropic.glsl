#ifndef SAMPLE_DISTANCE_ANISOTROPIC
#define SAMPLE_DISTANCE_ANISOTROPIC

// Samples the anisotropic distance texture at given coordinates and octant.
ivec3 sample_distance_anisotropic(in ivec3 block_coords, in int octant, out bool occupied)
{        
    ivec3 texel = block_coords + ivec3(0,0,octant * u_volume.blocked_dimensions.z);
    uint distance = texelFetch(u_textures.distance_map, texel, 0).r;
    occupied = (distance == 0u);
    return ivec3(distance);
}

#endif
