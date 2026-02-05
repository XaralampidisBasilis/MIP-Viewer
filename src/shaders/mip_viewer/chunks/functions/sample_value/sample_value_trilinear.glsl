// Samples the base volume using standard trilinear interpolation.
// Assumes texture uses linear filtering and normalized coordinates.
#ifndef SAMPLE_VALUE_TRILINEAR
#define SAMPLE_VALUE_TRILINEAR

float sample_value_trilinear(in vec3 coords)
{
    vec3 texture_coords = coords * u_volume.inv_dimensions;
    return texture(u_textures.interpolation_map, texture_coords).a;
}

float sample_residue_trilinear(in vec3 coords) 
{ 
    return sample_value_trilinear(coords) - u_volume.isovalue; 
}

#endif