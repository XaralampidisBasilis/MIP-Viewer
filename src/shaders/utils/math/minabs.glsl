#ifndef MINABS
#define MINABS

float minabs(in vec2 a)
{
    return abs(a.x) < abs(a.y) ? a.x : a.y;
}

float minabs(in vec3 a)
{
    float v = abs(a.x) < abs(a.y) ? a.x : a.y;
    return abs(v) < abs(a.z) ? v : a.z;
}

float minabs(in vec4 a)
{
    float v1 = abs(a.x) < abs(a.y) ? a.x : a.y;
    float v2 = abs(a.z) < abs(a.w) ? a.z : a.w;
    return abs(v1) < abs(v2) ? v1 : v2;
}

float minabs(in float a, in float b)
{
    return abs(a) < abs(b) ? a : b;
}

vec2 minabs(in vec2 a, in float b)
{
    return vec2(minabs(a.x, b), minabs(a.y, b));
}

vec3 minabs(in vec3 a, in float b)
{
    return vec3(minabs(a.x, b), minabs(a.y, b), minabs(a.z, b));
}

vec4 minabs(in vec4 a, in float b)
{
    return vec4(minabs(a.x, b), minabs(a.y, b), minabs(a.z, b), minabs(a.w, b));
}

vec2 minabs(in vec2 a, in vec2 b)
{
    return vec2(minabs(a.x, b.x), minabs(a.y, b.y));
}

vec3 minabs(in vec3 a, in vec3 b)
{
    return vec3(minabs(a.x, b.x), minabs(a.y, b.y), minabs(a.z, b.z));
}

vec4 minabs(in vec4 a, in vec4 b)
{
    return vec4(minabs(a.x, b.x), minabs(a.y, b.y), minabs(a.z, b.z), minabs(a.w, b.w));
}

#endif // MINABS
