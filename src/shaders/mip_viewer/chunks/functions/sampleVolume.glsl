// Samples the base volume using standard trilinear interpolation.
// Assumes texture uses linear filtering and normalized coordinates.
#ifndef SAMPLE_TRILINEAR_VOLUME
#define SAMPLE_TRILINEAR_VOLUME

float sampleVolume(vec3 position)
{
    vec3 texturePos = position * u_volume.inv_dimensions;
    return texture(u_textures.volume_map, texturePos).r;
}

#endif