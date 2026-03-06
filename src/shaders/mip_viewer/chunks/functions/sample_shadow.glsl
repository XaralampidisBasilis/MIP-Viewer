#ifndef SAMPLE_SHADOW
#define SAMPLE_SHADOW

// Samples the occupancy texture at the given integer coordinates.
bool sample_shadow(in ivec3 block_coords, in int ray_map)
{    
    int i = texelFetch(u_textures.shadow_map, block_coords, 0).r; // -2048..2047 half float precision 
    uint u = uint(i + 2048); // 0..4095
    uint s = (u >> ray_map) & 1u;

    return (s > 0u);
}

#endif