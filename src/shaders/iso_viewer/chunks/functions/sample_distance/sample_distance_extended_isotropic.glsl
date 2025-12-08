#ifndef SAMPLE_DISTANCE_EXTENDED_ISOTROPIC
#define SAMPLE_DISTANCE_EXTENDED_ISOTROPIC

#ifndef UNPACK_UINT5551
#include "../unpack_uint5551"
#endif

// Samples the isotropic distance texture at given integer coordinates.
void sample_distance_extended_isotropic(in ivec3 block_coords, out bool occupied, out ivec3 min_distances, out ivec3 max_distances)
{
    uvec2 packed = texelFetch(u_textures.distance_map, block_coords, 0).rg;
    uvec4 min_unpacked = unpack_uint5551(packed.r);
    uvec4 max_unpacked = unpack_uint5551(packed.g);
    min_distances = ivec3(min_unpacked.xyz);
    max_distances = ivec3(max_unpacked.xyz);
    occupied = bool(min_unpacked.w);
}

#endif
