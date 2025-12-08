/*
contributors: Patricio Gonzalez Vivo
description: extend GLSL min function to add more arguments
use:
    - min(<float> A, <float> B, <float> C[, <float> D])
    - min(<vec2|vec3|vec4> A)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef MMIN
#define MMIN

float mmin(in float a) { return a; }
float mmin(in float a,in float b) { return min(a, b); }
float mmin(in float a,in float b, in float c) { return min(a, min(b, c)); }
float mmin(in float a,in float b, in float c, in float d) { return min(min(a,b), min(c, d)); }

float mmin(vec2 v) { return min(v.x, v.y); }
float mmin(vec3 v) { return mmin(v.x, v.y, v.z); }
float mmin(vec4 v) { return mmin(v.x, v.y, v.z, v.w); }
float mmin(float v[5]) 
{
    float r = v[0];
    r = min(r, v[1]);
    r = min(r, v[2]);
    r = min(r, v[3]);
    r = min(r, v[4]);
    return r;
}
float mmin(float v[6]) 
{
    float r = v[0];
    r = min(r, v[1]);
    r = min(r, v[2]);
    r = min(r, v[3]);
    r = min(r, v[4]);
    r = min(r, v[5]);
    return r;
}


#endif // MMIN
