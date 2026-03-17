// Samples the base volume using standard trilinear interpolation.
// Assumes texture uses linear filtering and normalized coordinates.
#ifndef SAMPLE_TRILINEAR_VOLUME
#define SAMPLE_TRILINEAR_VOLUME

float sampleVolume(vec3 coords)
{
    vec3 texture_coords = coords * u_volume.inv_dimensions;
    return texture(u_textures.volume_map, texture_coords).r;
}

#endif