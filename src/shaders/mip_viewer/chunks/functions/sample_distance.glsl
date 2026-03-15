#ifndef SAMPLE_DISTANCE
#define SAMPLE_DISTANCE

// |  ray.map   |   ray.axis   |   ray.octant  |
// | ---------- | ------------ | ------------- |
// |          0 |      x       |      +++      |
// |          1 |      x       |      +-+      |
// |          2 |      x       |      ++-      |
// |          3 |      x       |      +--      |
// |          4 |      y       |      +++      |
// |          5 |      y       |      -++      |
// |          6 |      y       |      ++-      |
// |          7 |      y       |      -+-      |
// |          8 |      z       |      +++      |
// |          9 |      z       |      +-+      |
// |         10 |      z       |      -++      |
// |         11 |      z       |      --+      |

int sample_rgba16ui_distance(in ivec3 block_coords, out bool block_shadowed)
{    
    uvec4 u = texelFetch(u_textures.distance_map, block_coords, 0); 

    uvec4 x = (u >>  0) & 0x1fu;
    uvec4 y = (u >>  5) & 0x1fu;
    uvec4 z = (u >> 10) & 0x3fu;

    uint d = 0u;

         if (ray.map ==  0u) d = x[0]; // 'x', '+++'
    else if (ray.map ==  1u) d = x[1]; // 'x', '+-+'
    else if (ray.map ==  2u) d = x[2]; // 'x', '++-'
    else if (ray.map ==  3u) d = x[3]; // 'x', '+--'
    else if (ray.map ==  4u) d = y[0]; // 'y', '+++'
    else if (ray.map ==  5u) d = y[1]; // 'y', '-++'
    else if (ray.map ==  6u) d = y[2]; // 'y', '++-'
    else if (ray.map ==  7u) d = y[3]; // 'y', '-+-'
    else if (ray.map ==  8u) d = z[0]; // 'z', '+++'
    else if (ray.map ==  9u) d = z[1]; // 'z', '+-+'
    else if (ray.map == 10u) d = z[2]; // 'z', '-++'
    else if (ray.map == 11u) d = z[3]; // 'z', '--+'

    block_shadowed = (d != 0u);
    return int(max(d, 1u));
}

int sample_rgb32ui_distance(in ivec3 block_coords, out bool block_shadowed)
{    
    uvec3 u = texelFetch(u_textures.distance_map, block_coords, 0).rgb; 

    uvec3 u0 = (u >>  0) & 0xffu;
    uvec3 u1 = (u >>  8) & 0xffu;
    uvec3 u2 = (u >> 16) & 0xffu;
    uvec3 u3 = (u >> 24) & 0xffu;

    uint d = 0u;

         if (ray.map ==  0u) d = u0.z; // 'x', '+++'
    else if (ray.map ==  1u) d = u1.z; // 'x', '+-+'
    else if (ray.map ==  2u) d = u2.z; // 'x', '++-'
    else if (ray.map ==  3u) d = u3.z; // 'x', '+--'
    else if (ray.map ==  4u) d = u0.y; // 'y', '+++'
    else if (ray.map ==  5u) d = u1.y; // 'y', '-++'
    else if (ray.map ==  6u) d = u2.y; // 'y', '++-'
    else if (ray.map ==  7u) d = u3.y; // 'y', '-+-'
    else if (ray.map ==  8u) d = u0.z; // 'z', '+++'
    else if (ray.map ==  9u) d = u1.z; // 'z', '+-+'
    else if (ray.map == 10u) d = u2.z; // 'z', '-++'
    else if (ray.map == 11u) d = u3.z; // 'z', '--+'

    block_shadowed = (d != 0u);
    return int(max(d, 1u));
}

int sample_rgba16ui_distance_fast(in ivec3 block_coords, out bool block_shadowed)
{
    uvec4 u = texelFetch(u_textures.distance_map, block_coords, 0);

    uint shift = ray.axis * 5u;
    uint mask  = (ray.axis == 2u) ? 0x3Fu : 0x1Fu;

    uint d = (u[ray.idx] >> shift) & mask;

    block_shadowed = (d != 0u);
    return int(max(d, 1u));
}

int sample_rgb32ui_distance_fast(in ivec3 block_coords, out bool block_shadowed)
{
    uvec3 u = texelFetch(u_textures.distance_map, block_coords, 0).rgb;

    uint shift = ray.idx * 8u;
    uint mask = 0xFFu;

    uint d = (u[ray.axis] >> shift) & mask;

    block_shadowed = (d != 0u);
    return int(max(d, 1u));
}


#endif

