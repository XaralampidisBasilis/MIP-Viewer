#ifndef SIGN_CHANGES
#define SIGN_CHANGES

int sign_changes(float a, float b) 
{
    return int((a < 0.0) != (b < 0.0));
}

int sign_changes(vec2 v) 
{
    return int((v.x < 0.0) != (v.y < 0.0));
}

int sign_changes(vec3 v) 
{
    return int((v.x < 0.0) != (v.y < 0.0)) +
           int((v.y < 0.0) != (v.z < 0.0));
}

int sign_changes(vec4 v) 
{
    return int((v.x < 0.0) != (v.y < 0.0)) +
           int((v.y < 0.0) != (v.z < 0.0)) +
           int((v.z < 0.0) != (v.w < 0.0));
}

int sign_changes(float v[5]) 
{
    return int((v[0] < 0.0) != (v[1] < 0.0)) +
           int((v[1] < 0.0) != (v[2] < 0.0)) +
           int((v[2] < 0.0) != (v[3] < 0.0)) +
           int((v[3] < 0.0) != (v[4] < 0.0));
}

int sign_changes(float v[6]) 
{
    return int((v[0] < 0.0) != (v[1] < 0.0)) +
           int((v[1] < 0.0) != (v[2] < 0.0)) +
           int((v[2] < 0.0) != (v[3] < 0.0)) +
           int((v[3] < 0.0) != (v[4] < 0.0)) +
           int((v[4] < 0.0) != (v[5] < 0.0));
}

#endif
