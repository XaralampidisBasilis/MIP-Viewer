#ifndef UNIFORMS_TEXTURES
#define UNIFORMS_TEXTURES

struct UniformsTextures 
{
    sampler3D volume_map;
    isampler3D shadow_map;
    // usampler3D distance_map;
};

uniform UniformsTextures u_textures;

#endif