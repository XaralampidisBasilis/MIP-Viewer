#ifndef SAMPLE_SHADOW
#define SAMPLE_SHADOW

bool sampleShadow1bit(in ivec3 coords)
{    
    uint u = texelFetch(u_textures.distance_map, coords, 0).r; // 0..4095 
    uint d = (u >> u_ray.map) & 0x1u;

    return (d != 0u);
}

bool sampleShadow5bit(in ivec3 coords)
{
    uvec4 u = texelFetch(u_textures.distance_map, coords, 0).rgba;

    uint shift = u_ray.axis * 5u;
    uint mask  = (u_ray.axis == 2u) ? 0x3Fu : 0x1Fu;

    uint d = (u[u_ray.idx] >> shift) & mask;

    return (d != 0u);
}

bool sampleShadow8bit(in ivec3 coords)
{
    uvec3 u = texelFetch(u_textures.distance_map, coords, 0).rgb;

    uint shift = u_ray.idx * 8u;
    uint mask = 0xFFu;

    uint d = (u[u_ray.axis] >> shift) & mask;

    return (d != 0u);
}

bool sampleShadow10bit(in ivec3 coords)
{
    uvec4 u = texelFetch(u_textures.distance_map, coords, 0).rgba;

    const uint MASKS[3]  = uint[3](0x7FFu, 0x7FFu, 0x3FFu);
    uint shift = u_ray.axis *11u;

    uint packed = u[u_ray.idx];
    uint d = (packed >> shift) & MASKS[u_ray.axis];

    return (d != 0u);
}

#endif
