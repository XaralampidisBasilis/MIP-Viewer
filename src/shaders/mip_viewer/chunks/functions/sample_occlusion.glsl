#ifndef SAMPLE_OCCLUSION
#define SAMPLE_OCCLUSION

// Samples the occupancy texture at the given integer coordinates.
bool sample_occlusion(in ivec3 block_coords, in int map)
{    
    int i = texelFetch(u_textures.occlusion_map, block_coords, 0).r; // -2048..2047 half float precision 
    uint u = uint(i + 2048); // 0..4095
    uint o = (u >> map) & 1u;

    return (o > 0u);
}

#endif