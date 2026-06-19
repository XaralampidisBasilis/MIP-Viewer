#ifndef ARGMAX
#define ARGMAX

// int argmax(in float x) 
// { 
//     return 0; 
// }

// int argmax(in vec2 v) 
// { 
//     return int(v.x < v.y); 
// }

// int argmax(in vec3 v) 
// { 
//     return (v.x >= v.y) 
//         ? ((v.x >= v.z) ? 0 : 2)
//         : ((v.y >= v.z) ? 1 : 2);
// }

// int argmax(in vec4 v) 
// { 
//     int i = 0;
//     float m = v.x;
//     if (v.y > m) { i = 1; m = v.y; }
//     if (v.z > m) { i = 2; m = v.z; }
//     if (v.w > m) { i = 3; }
//     return i;
// }

int argmax(in float x) 
{ 
    return 0; 
}

int argmax(in vec2 v) 
{ 
    return int(v.x < v.y); 
}

int argmax(in vec3 v) 
{ 
    int i = int(v.x < v.y);
    i += int(v[i] < v.z) * (2 - i);
    return i;
}

int argmax(in vec4 v) 
{ 
    int i = int(v.x < v.y);
    i += int(v[i] < v.z) * (2 - i);
    i += int(v[i] < v.w) * (3 - i);
    return i;
}

int argmax(in float x, in float y) 
{ 
    return argmax(vec2(x, y)); 
}

int argmax(in float x, in float y, in float z) 
{ 
    return argmax(vec3(x, y, z));
}

int argmax(in float x, in float y, in float z, in float w) 
{ 
    return argmax(vec4(x, y, z, w));
}

#endif 
