#ifndef SAMPLE_DISTANCE
#define SAMPLE_DISTANCE

// |  ray.map   |   ray.axis   |        ray.octants        |
// | ---------- | ------------ | ------------------------- |
// |          0 |      x       |        ---, +++           |
// |          1 |      x       |        -+-, +-+           |
// |          2 |      x       |        ++-, --+           |
// |          3 |      x       |        +--, -++           |
// |          4 |      y       |        ---, +++           |
// |          5 |      y       |        +--, -++           |
// |          6 |      y       |        ++-, --+           |
// |          7 |      y       |        -+-, +-+           |
// |          8 |      z       |        ---, +++           |
// |          9 |      z       |        -+-, +-+           |
// |         10 |      z       |        +--, -++           |
// |         11 |      z       |        ++-, --+           |


// int sample_distance(in ivec3 block_coords, in int ray_map, out bool block_shadowed)
// {    
//     uvec4 u = texelFetch(u_textures.distance_map, block_coords, 0); 

//     uvec4 x = (u >> 11) & 0x1Fu;
//     uvec4 y = (u >>  5) & 0x1Fu;
//     uvec4 z = (u >>  0) & 0x1Fu;

//     uint d = 0u;

//          if (ray_map ==  0) d = x[0]; // 'x', '+++'
//     else if (ray_map ==  1) d = x[1]; // 'x', '+-+'
//     else if (ray_map ==  2) d = x[2]; // 'x', '++-'
//     else if (ray_map ==  3) d = x[3]; // 'x', '+--'
//     else if (ray_map ==  4) d = y[0]; // 'y', '+++'
//     else if (ray_map ==  5) d = y[1]; // 'y', '-++'
//     else if (ray_map ==  6) d = y[2]; // 'y', '++-'
//     else if (ray_map ==  7) d = y[3]; // 'y', '-+-'
//     else if (ray_map ==  8) d = z[0]; // 'z', '+++'
//     else if (ray_map ==  9) d = z[1]; // 'z', '+-+'
//     else if (ray_map == 10) d = z[2]; // 'z', '-++'
//     else if (ray_map == 11) d = z[3]; // 'z', '--+'

//     block_shadowed = (d != 0u);
//     return int(max(d, 1u));
// }

int sample_distance(in ivec3 block_coords, in int ray_map, out bool block_shadowed)
{
    uvec4 u = texelFetch(u_textures.distance_map, block_coords, 0);

    uint map  = uint(ray_map);
    uint idx  = map & 3u;       // 0..3
    uint axis = map >> 2u;      // 0=x, 1=y, 2=z

    const uint lut[3] = uint[3](11u, 5u, 0u);
    uint shift = lut[axis];

    uint d = (u[idx] >> shift) & 0x1Fu;

    block_shadowed = (d != 0u);
    return int(max(d, 1u));
}


#endif

