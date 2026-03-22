#ifndef SAMPLE_SHADOW
#define SAMPLE_SHADOW

// Samples the occupancy texture at the given integer coordinates.
bool sampleShadow(in ivec3 coords)
{    
    int i = texelFetch(u_textures.shadow_map, coords, 0).r; // -2048..2047 half float precision     
    uint u = uint(i + 2048); // 0..4095
    uint s = (u >> u_ray.map) & 0x1u;

    return (s == 1u);
}

#endif
