
#ifndef ABS_L1_NORMALIZATION
#define ABS_L1_NORMALIZATION

vec4 abs_l1_normalization(vec4 v) 
{
    vec4 u = abs(v);
    float s = u.x + u.y + u.z + u.w;   
    return (s > 0.0) ? (u / s) : vec4(0.0);
}

void abs_l1_normalization(in float v[6], out float u[6]) 
{
    float s = 0.0;

    for (int i = 0; i < 6; ++i) 
    {
        u[i] = abs(v[i]);
        s += u[i];
    }

    float inv = (s > 0.0) ? 1.0 / s : 0.0;
    for (int i = 0; i < 6; ++i) 
    {
        u[i] *= inv;
    }
}

#endif