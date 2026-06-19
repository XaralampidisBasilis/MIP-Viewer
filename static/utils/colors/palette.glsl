/*
https://iquilezles.org/articles/palettes/
*/
#ifndef PALETTE
#define PALETTE

// cosine based palette, 4 vec3 params
vec3 palette( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d )
{
    return a + b*cos( 6.283185*(c*t+d) );
}

vec3 palette(float t, vec3 a, vec3 b0, vec3 c0, vec3 d0, vec3 b1, vec3 c1, vec3 d1)
{
    return a + b0 * cos(6.283185 * (c0 * t + d0)) + b1 * cos(6.283185 * (c1 * t + d1));
}


#endif
