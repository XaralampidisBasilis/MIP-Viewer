#ifndef ARGMIN
#define ARGMIN

// int argmin(in float x) 
// { 
//     return 0; 
// }

// int argmin(in vec2 v) 
// { 
//     return int(v.x > v.y); 
// }

// int argmin(in vec3 v) 
// { 
//     return (v.x <= v.y) 
//         ? ((v.x <= v.z) ? 0 : 2)
//         : ((v.y <= v.z) ? 1 : 2);
// }

// int argmin(in vec4 v) 
// { 
//     int i = 0;
//     float m = v.x;
//     if (v.y < m) { i = 1; m = v.y; }
//     if (v.z < m) { i = 2; m = v.z; }
//     if (v.w < m) { i = 3; }
//     return i;
// }

int argmin(in float x) 
{ 
    return 0; 
}

int argmin(in vec2 v) 
{ 
    return int(v.x > v.y); 
}

int argmin(in vec3 v) 
{ 
    int i = int(v.x > v.y);
    i += int(v[i] > v.z) * (2 - i);
    return i;
}

int argmin(in vec4 v) 
{ 
    int i = int(v.x > v.y);
    i += int(v[i] > v.z) * (2 - i);
    i += int(v[i] > v.w) * (3 - i);
    return i;
}

int argmin(in float x, in float y) 
{ 
    return argmin(vec2(x, y)); 
}

int argmin(in float x, in float y, in float z) 
{ 
    return argmin(vec3(x, y, z));
}

int argmin(in float x, in float y, in float z, in float w) 
{ 
    return argmin(vec4(x, y, z, w));
}


#endif 
