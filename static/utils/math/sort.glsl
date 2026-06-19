#ifndef SORT
#define SORT

#ifndef SORT_MAX_LENGTH
#define SORT_MAX_LENGTH 5
#endif

void sort(inout vec2 v)
{
    v =  (v.x > v.y) ? v.yx : v.xy;
}

void sort(inout vec3 v)
{
    v.xy = (v.x > v.y) ? v.yx : v.xy;
    v.yz = (v.y > v.z) ? v.zy : v.yz;
    v.xy = (v.x > v.y) ? v.yx : v.xy;
}

// This uses Bose–Nelson sorting network for 4 inputs
void sort(inout vec4 v)
{
    v.xy = (v.x > v.y) ? v.yx : v.xy;
    v.zw = (v.z > v.w) ? v.wz : v.zw;
    v.xz = (v.x > v.z) ? v.zx : v.xz;
    v.yw = (v.y > v.w) ? v.wy : v.yw;
    v.yz = (v.y > v.z) ? v.zy : v.yz;
}

void sort(inout float v[SORT_MAX_LENGTH])
{
    float t; 

    for (int i = 0; i < SORT_MAX_LENGTH - 1; ++i)
    {
        for (int j = 0; j < SORT_MAX_LENGTH - 1 - i; ++j)
        {
            if (v[j] > v[j + 1])
            {
                t = v[j];
                v[j] = v[j + 1];
                v[j + 1] = t;
            }
        }
    }
}

void sort(inout vec2 v, inout vec2 u)
{
    bool c = (v.x > v.y);
    v = c ? v.yx : v.xy;
    u = c ? u.yx : u.xy;
}

void sort(inout vec3 v, inout vec3 u)
{
    bool c = (v.x > v.y);
    v.xy = c ? v.yx : v.xy;
    u.xy = c ? u.yx : u.xy;

    c = (v.y > v.z);
    v.yz = c ? v.zy : v.yz;
    u.yz = c ? u.zy : u.yz;

    c = (v.x > v.y);
    v.xy = c ? v.yx : v.xy;
    u.xy = c ? u.yx : u.xy;
}

void sort(inout vec4 v, inout vec4 u)
{
    bool c = (v.x > v.y);
    v.xy = c ? v.yx : v.xy;
    u.xy = c ? u.yx : u.xy;

    c = (v.z > v.w);
    v.zw = c ? v.wz : v.zw;
    u.zw = c ? u.wz : u.zw;

    c = (v.x > v.z);
    v.xz = c ? v.zx : v.xz;
    u.xz = c ? u.zx : u.xz;

    c = (v.y > v.w);
    v.yw = c ? v.wy : v.yw;
    u.yw = c ? u.wy : u.yw;

    c = (v.y > v.z);
    v.yz = c ? v.zy : v.yz;
    u.yz = c ? u.zy : u.yz;
}

void sort(inout float v[SORT_MAX_LENGTH], inout float u[SORT_MAX_LENGTH])
{
    float t; 

    for (int i = 0; i < SORT_MAX_LENGTH - 1; ++i)
    {
        for (int j = 0; j < SORT_MAX_LENGTH - 1 - i; ++j)
        {
            if (v[j] > v[j + 1])
            {
                // Swap v
                t = v[j];
                v[j] = v[j + 1];
                v[j + 1] = t;

                // Swap u in sync
                t = u[j];
                u[j] = u[j + 1];
                u[j + 1] = t;
            }
        }
    }
}



#endif // SORT
