#ifndef SORT
#define SORT

// n-element Bose–Nelson networks

void sort(inout float a, inout float b) 
{
    float x = min(a, b); 
    float y = max(a, b); 
    a = x; 
    b = y;
}

void sort(inout float a, inout float b, inout float c) 
{
    sort(a, b);
    sort(b, c);
    sort(a, b);
}

void sort(inout float a, inout float b, inout float c, inout float d) 
{
    sort(a, b);
    sort(c, d);
    sort(a, c);
    sort(b, d);
    sort(b, c);
}

void sort(inout vec2 v)
{
    float x = min(v.x, v.y); 
    float y = max(v.x, v.y); 
    v.x = x; 
    v.y = y;
}

void sort(inout vec3 v) 
{
    sort(v.xy);
    sort(v.yz);
    sort(v.xy);
}

void sort(inout vec4 v) 
{
    sort(v.xy);
    sort(v.zw);
    sort(v.xz);
    sort(v.yw);
    sort(v.yz);
}

#endif // SORT
