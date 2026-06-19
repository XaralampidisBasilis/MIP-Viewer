#ifndef MAXABS
#define MAXABS


float maxabs(in vec2 a) 
{
    return abs(a.x) > abs(a.y) ? a.x : a.y;
}

float maxabs(in vec3 a) 
{
    float v = abs(a.x) > abs(a.y) ? a.x : a.y;
    return abs(v) > abs(a.z) ? v : a.z;
}

float maxabs(in vec4 a) 
{
    float v1 = abs(a.x) > abs(a.y) ? a.x : a.y;
    float v2 = abs(a.z) > abs(a.w) ? a.z : a.w;
    return abs(v1) > abs(v2) ? v1 : v2;
}

float maxabs(in float a, in float b) 
{
    return abs(a) > abs(b) ? a : b;
}

vec2 maxabs(in vec2 a, in float b) 
{
    return vec2(maxabs(a.x, b), maxabs(a.y, b));
}

vec3 maxabs(in vec3 a, in float b) 
{
    return vec3(maxabs(a.x, b), maxabs(a.y, b), maxabs(a.z, b));
}

vec4 maxabs(in vec4 a, in float b) 
{
    return vec4(maxabs(a.x, b), maxabs(a.y, b), maxabs(a.z, b), maxabs(a.w, b));
}

vec2 maxabs(in vec2 a, in vec2 b) 
{
    return vec2(maxabs(a.x, b.x), maxabs(a.y, b.y));
}

vec3 maxabs(in vec3 a, in vec3 b) 
{
    return vec3(maxabs(a.x, b.x), maxabs(a.y, b.y), maxabs(a.z, b.z));
}

vec4 maxabs(in vec4 a, in vec4 b) 
{
    return vec4(maxabs(a.x, b.x), maxabs(a.y, b.y), maxabs(a.z, b.z), maxabs(a.w, b.w));
}

#endif // MAXABS
