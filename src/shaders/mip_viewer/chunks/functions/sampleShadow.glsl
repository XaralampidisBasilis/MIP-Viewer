#ifndef SAMPLE_SHADOW
#define SAMPLE_SHADOW


bool sampleShadow1bit(in ivec3 coords)
{    
    uvec4 u = texelFetch(u_textures.distance_map, coords, 0); 
    uint packed = u.r;

    uint mask = 0x1u;
    uint shift = u_ray.group_index;

    uint d = (packed >> shift) & mask;
    return (d != 0u);
}

bool sampleShadow5bit(in ivec3 coords)
{
    uvec4 u = texelFetch(u_textures.distance_map, coords, 0);
    uint packed = u[u_ray.quadrant_index];

    const uint MASKS[3] = uint[3](0x1Fu, 0x1Fu, 0x3Fu);
    uint mask = MASKS[u_ray.dominant_axis];
    uint shift = u_ray.dominant_axis * 5u;

    uint d = (packed >> shift) & mask;
    return (d != 0u);
}

bool sampleShadow8bit(in ivec3 coords)
{
    uvec4 u = texelFetch(u_textures.distance_map, coords, 0);
    uint packed = u[u_ray.dominant_axis];

    uint shift = u_ray.quadrant_index * 8u;
    uint mask = 0xFFu;

    uint d = (packed >> shift) & mask;
    return (d != 0u);
}

bool sampleShadow10bit(in ivec3 coords)
{
    uvec4 u = texelFetch(u_textures.distance_map, coords, 0);
    uint packed = u[u_ray.quadrant_index];

    const uint MASKS[3]  = uint[3](0x7FFu, 0x7FFu, 0x3FFu);
    uint shift = u_ray.dominant_axis *11u;
    uint mask =  MASKS[u_ray.dominant_axis];

    uint d = (packed >> shift) & mask;
    return (d != 0u);
}

bool sampleShadow(in ivec3 coords)
{
    uvec4 u = texelFetch(u_textures.distance_map, coords, 0);

    #if DISTANCE_VARIATION == 0 

        uint packed = u.r;
        uint shift = u_ray.group_index;
        uint mask = 0x1u;

    #elif DISTANCE_VARIATION == 1

        const uint MASKS[3] = uint[3](0x1Fu, 0x1Fu, 0x3Fu);

        uint packed = u[u_ray.quadrant_index];
        uint shift = u_ray.dominant_axis * 5u;
        uint mask = MASKS[u_ray.dominant_axis];

    #elif DISTANCE_VARIATION == 2 

        uint packed = u[u_ray.dominant_axis];
        uint shift = u_ray.quadrant_index * 8u;
        uint mask = 0xFFu;

    #elif DISTANCE_VARIATION == 3 

        const uint MASKS[3]  = uint[3](0x7FFu, 0x7FFu, 0x3FFu);

        uint packed = u[u_ray.quadrant_index];
        uint shift = u_ray.dominant_axis * 11u;
        uint mask = MASKS[u_ray.dominant_axis];

    #endif

    uint d = (packed >> shift) & mask;
    return (d != 0u);
}

/*

bool sampleShadow(in ivec3 coords)
{
    #if DISTANCE_VARIATION == 0 
    return sampleShadow1bit(coords);

    #elif DISTANCE_VARIATION == 1 
    return sampleShadow5bit(coords);

    #elif DISTANCE_VARIATION == 2 
    return sampleShadow8bit(coords);

    #elif DISTANCE_VARIATION == 3 
    return sampleShadow10bit(coords);

    #endif
}

bool sampleShadow(in ivec3 coords)
{
    uvec4 u = texelFetch(u_textures.distance_map, coords, 0);

    #if DISTANCE_VARIATION == 0 

        uint packed = u.r;
        uint shift = u_ray.group_index;
        uint mask = 0x1u;

    #elif DISTANCE_VARIATION == 1

        const uint MASKS[3] = uint[3](0x1Fu, 0x1Fu, 0x3Fu);

        uint packed = u[u_ray.quadrant_index];
        uint shift = u_ray.dominant_axis * 5u;
        uint mask = MASKS[u_ray.dominant_axis];

    #elif DISTANCE_VARIATION == 2 

        uint packed = u[u_ray.dominant_axis];
        uint shift = u_ray.quadrant_index * 8u;
        uint mask = 0xFFu;

    #elif DISTANCE_VARIATION == 3 

        const uint MASKS[3]  = uint[3](0x7FFu, 0x7FFu, 0x3FFu);

        uint packed = u[u_ray.quadrant_index];
        uint shift = u_ray.dominant_axis * 11u;
        uint mask = MASKS[u_ray.dominant_axis];

    #endif

    uint d = (packed >> shift) & mask;
    return (d != 0u);
}
*/

#endif
