#ifndef SAMPLE_DISTANCE
#define SAMPLE_DISTANCE

int sampleDistance1bit(in ivec3 coords, out bool empty)
{    
    uint u = texelFetch(u_textures.distance_map, coords, 0).r; // 0..4095 
    uint d = (u >> u_ray.map) & 0x1u;

    empty = (d != 0u);
    return 1;
}

int sampleDistance5bit(in ivec3 coords, out bool empty)
{
    uvec4 u = texelFetch(u_textures.distance_map, coords, 0);

    const uint MASKS[3] = uint[3](0x1Fu, 0x1Fu, 0x3Fu);
    uint shift = u_ray.axis * 5u;

    uint d = (u[u_ray.idx] >> shift) & MASKS[u_ray.axis];

    empty = (d != 0u);
    return int(max(d, 1u));
}

int sampleDistance10bit(in ivec3 coords, out bool empty)
{
    uvec4 u = texelFetch(u_textures.distance_map, coords, 0).rgba;

    const uint MASKS[3]  = uint[3](0x7FFu, 0x7FFu, 0x3FFu);
    uint shift = u_ray.axis *11u;

    uint packed = u[u_ray.idx];
    uint d = (packed >> shift) & MASKS[u_ray.axis];

    empty = (d != 0u);
    return int(max(d, 1u));
}

/* explicit versions to understand logic

void sampleDistance1bit(in ivec3 coords, out bool empty)
{    
    uint u = texelFetch(u_textures.distance_map, coords, 0).r; // 0..4095 
    uint d = 0u;

         if (u_ray.map ==  0u) d = (u >>  0) & 0x1u; // 'x', '+++'
    else if (u_ray.map ==  1u) d = (u >>  1) & 0x1u; // 'x', '+-+'
    else if (u_ray.map ==  2u) d = (u >>  2) & 0x1u; // 'x', '++-'
    else if (u_ray.map ==  3u) d = (u >>  3) & 0x1u; // 'x', '+--'
    else if (u_ray.map ==  4u) d = (u >>  4) & 0x1u; // 'y', '+++'
    else if (u_ray.map ==  5u) d = (u >>  5) & 0x1u; // 'y', '-++'
    else if (u_ray.map ==  6u) d = (u >>  6) & 0x1u; // 'y', '++-'
    else if (u_ray.map ==  7u) d = (u >>  7) & 0x1u; // 'y', '-+-'
    else if (u_ray.map ==  8u) d = (u >>  8) & 0x1u; // 'z', '+++'
    else if (u_ray.map ==  9u) d = (u >>  9) & 0x1u; // 'z', '+-+'
    else if (u_ray.map == 10u) d = (u >> 10) & 0x1u; // 'z', '-++'
    else if (u_ray.map == 11u) d = (u >> 11) & 0x1u; // 'z', '--+'

    empty = (d == 1u);
}

int sampleDistance5bit(in ivec3 coords, out bool empty)
{    
    uvec4 u = texelFetch(u_textures.distance_map, coords, 0); 

    uvec4 x = (u >>  0) & 0x1fu;
    uvec4 y = (u >>  5) & 0x1fu;
    uvec4 z = (u >> 10) & 0x3fu;

    uint d = 0u;

         if (u_ray.map ==  0u) d = x[0]; // 'x', '+++'
    else if (u_ray.map ==  1u) d = x[1]; // 'x', '+-+'
    else if (u_ray.map ==  2u) d = x[2]; // 'x', '++-'
    else if (u_ray.map ==  3u) d = x[3]; // 'x', '+--'
    else if (u_ray.map ==  4u) d = y[0]; // 'y', '+++'
    else if (u_ray.map ==  5u) d = y[1]; // 'y', '-++'
    else if (u_ray.map ==  6u) d = y[2]; // 'y', '++-'
    else if (u_ray.map ==  7u) d = y[3]; // 'y', '-+-'
    else if (u_ray.map ==  8u) d = z[0]; // 'z', '+++'
    else if (u_ray.map ==  9u) d = z[1]; // 'z', '+-+'
    else if (u_ray.map == 10u) d = z[2]; // 'z', '-++'
    else if (u_ray.map == 11u) d = z[3]; // 'z', '--+'

    empty = (d != 0u);
    return int(max(d, 1u));
}

int sampleDistance8bit(in ivec3 coords, out bool empty)
{    
    uvec3 u = texelFetch(u_textures.distance_map, coords, 0).rgb; 

    uvec3 u0 = (u >>  0) & 0xffu;
    uvec3 u1 = (u >>  8) & 0xffu;
    uvec3 u2 = (u >> 16) & 0xffu;
    uvec3 u3 = (u >> 24) & 0xffu;

    uint d = 0u;

         if (u_ray.map ==  0u) d = u0.z; // 'x', '+++'
    else if (u_ray.map ==  1u) d = u1.z; // 'x', '+-+'
    else if (u_ray.map ==  2u) d = u2.z; // 'x', '++-'
    else if (u_ray.map ==  3u) d = u3.z; // 'x', '+--'
    else if (u_ray.map ==  4u) d = u0.y; // 'y', '+++'
    else if (u_ray.map ==  5u) d = u1.y; // 'y', '-++'
    else if (u_ray.map ==  6u) d = u2.y; // 'y', '++-'
    else if (u_ray.map ==  7u) d = u3.y; // 'y', '-+-'
    else if (u_ray.map ==  8u) d = u0.z; // 'z', '+++'
    else if (u_ray.map ==  9u) d = u1.z; // 'z', '+-+'
    else if (u_ray.map == 10u) d = u2.z; // 'z', '-++'
    else if (u_ray.map == 11u) d = u3.z; // 'z', '--+'

    empty = (d != 0u);
    return int(max(d, 1u));
}
*/

#endif

