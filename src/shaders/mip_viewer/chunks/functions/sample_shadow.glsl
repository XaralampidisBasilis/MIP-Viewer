#ifndef SAMPLE_SHADOW
#define SAMPLE_SHADOW

// Samples the occupancy texture at the given integer coordinates.
bool sample_shadow(in ivec3 block_coords)
{    
    int i = texelFetch(u_textures.shadow_map, block_coords, 0).r; // -2048..2047 half float precision     
    uint u = uint(i + 2048); // 0..4095
    uint s = (u >> u_ray.map) & 0x1u;

    return (s == 1u);
}

bool sample_shadow_rgba16ui(in ivec3 block_coords)
{
    uvec4 u = texelFetch(u_textures.distance_map, block_coords, 0);

    uint shift = u_ray.axis * 5u;
    uint mask  = (u_ray.axis == 2u) ? 0x3Fu : 0x1Fu;

    uint d = (u[u_ray.idx] >> shift) & mask;
    return (d != 0u);
}

bool sample_shadow_rgb32ui(in ivec3 block_coords)
{
    uvec3 u = texelFetch(u_textures.distance_map, block_coords, 0).rgb;

    uint shift = u_ray.idx * 8u;
    uint mask = 0xFFu;

    uint d = (u[u_ray.axis] >> shift) & mask;
    return (d != 0u);
}


#endif
