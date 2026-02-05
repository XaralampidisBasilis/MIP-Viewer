#ifndef SAMPLE_DISTANCE_EXTENDED
#define SAMPLE_DISTANCE_EXTENDED

#ifndef UNPACK_UINT5551
#include "../unpack_uint5551"
#endif

// Samples the extended distance texture at given coordinates and octant.
// Returns 3-component distance vector and sets occupancy flag.
ivec3 sample_distance_extended_anisotropic(in ivec3 block_coords, in int octant, out bool occupied)
{
    ivec3 texel = block_coords + ivec3(0,0,octant * u_volume.blocked_dimensions.z);
    uint  packed = texelFetch(u_textures.distance_map, texel, 0).r;
    uvec4 unpacked = unpack_uint5551(packed);
    ivec3 distances = ivec3(unpacked.xyz);
    occupied = bool(unpacked.w);
    return distances;
}

#endif
