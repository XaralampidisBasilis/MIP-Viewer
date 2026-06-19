#ifndef SIGN_CHANGE
#define SIGN_CHANGE

bool sign_change(float a, float b) 
{
    return (a < 0.0) != (b < 0.0);
}

bool sign_change(vec2 v) 
{
    return (v.x < 0.0) != (v.y < 0.0);
}

bool sign_change(vec3 v) 
{
    return (v.x < 0.0) != (v.y < 0.0) ||
           (v.y < 0.0) != (v.z < 0.0);
}

bool sign_change(vec4 v) 
{
    return (v.x < 0.0) != (v.y < 0.0) ||
           (v.y < 0.0) != (v.z < 0.0) ||
           (v.z < 0.0) != (v.w < 0.0);
}

bool sign_change(float v[5]) 
{
    return (v[0] < 0.0) != (v[1] < 0.0) ||
           (v[1] < 0.0) != (v[2] < 0.0) ||
           (v[2] < 0.0) != (v[3] < 0.0) ||
           (v[3] < 0.0) != (v[4] < 0.0);
}

bool sign_change(float v[6]) 
{
    return (v[0] < 0.0) != (v[1] < 0.0) ||
           (v[1] < 0.0) != (v[2] < 0.0) ||
           (v[2] < 0.0) != (v[3] < 0.0) ||
           (v[3] < 0.0) != (v[4] < 0.0) ||
           (v[4] < 0.0) != (v[5] < 0.0);
}

#endif