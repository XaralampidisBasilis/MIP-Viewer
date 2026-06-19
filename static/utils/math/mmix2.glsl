// Bilinear mix

#ifndef MMIX2
#define MMIX2

#ifndef MMIX
#include "./mmix"
#endif

float mmix2(float a00, float a10, float a01, float a11, vec2 pct) 
{
    float a0 = mix(a00, a10, pct.x);
    float a1 = mix(a01, a11, pct.x);
    return mix(a0, a1, pct.y);
}

vec2 mmix2(vec2 a00, vec2 a10, vec2 a01, vec2 a11, vec2 pct) 
{
    vec2 a0 = mix(a00, a10, pct.x);
    vec2 a1 = mix(a01, a11, pct.x);
    return mix(a0, a1, pct.y);
}

vec3 mmix2(vec3 a00, vec3 a10, vec3 a01, vec3 a11, vec2 pct) 
{
    vec3 a0 = mix(a00, a10, pct.x);
    vec3 a1 = mix(a01, a11, pct.x);
    return mix(a0, a1, pct.y);
}

vec4 mmix2(vec4 a00, vec4 a10, vec4 a01, vec4 a11, vec2 pct) 
{
    vec4 a0 = mix(a00, a10, pct.x);
    vec4 a1 = mix(a01, a11, pct.x);
    return mix(a0, a1, pct.y);
}

float mmix2(float a00, float a10, float a20, float a01, float a11, float a21, float a02, float a12, float a22, vec2 pct) 
{
    float a0 = mmix(a00, a10, a20, pct.x);
    float a1 = mmix(a01, a11, a21, pct.x);
    float a2 = mmix(a02, a12, a22, pct.x);
    return mmix(a0, a1, a2, pct.y);
}
vec2 mmix2(vec2 a00, vec2 a10, vec2 a20, vec2 a01, vec2 a11, vec2 a21, vec2 a02, vec2 a12, vec2 a22, vec2 pct) 
{
    vec2 a0 = mmix(a00, a10, a20, pct.x);
    vec2 a1 = mmix(a01, a11, a21, pct.x);
    vec2 a2 = mmix(a02, a12, a22, pct.x);
    return mmix(a0, a1, a2, pct.y);
}
vec3 mmix2(vec3 a00, vec3 a10, vec3 a20, vec3 a01, vec3 a11, vec3 a21, vec3 a02, vec3 a12, vec3 a22, vec2 pct) 
{
    vec3 a0 = mmix(a00, a10, a20, pct.x);
    vec3 a1 = mmix(a01, a11, a21, pct.x);
    vec3 a2 = mmix(a02, a12, a22, pct.x);
    return mmix(a0, a1, a2, pct.y);
}
vec4 mmix2(vec4 a00, vec4 a10, vec4 a20, vec4 a01, vec4 a11, vec4 a21, vec4 a02, vec4 a12, vec4 a22, vec2 pct) 
{
    vec4 a0 = mmix(a00, a10, a20, pct.x);
    vec4 a1 = mmix(a01, a11, a21, pct.x);
    vec4 a2 = mmix(a02, a12, a22, pct.x);
    return mmix(a0, a1, a2, pct.y);
}

#endif // MMIX2