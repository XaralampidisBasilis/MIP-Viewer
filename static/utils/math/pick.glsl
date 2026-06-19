#ifndef PICK
#define PICK

float pick(in bool cond, in float a, in float b) 
{ 
    return (cond ? a : b); 
}

vec2 pick(in bvec2 cond, in vec2 a, in vec2 b) 
{ 
    return vec2(
        (cond.x ? a.x : b.x),
        (cond.y ? a.y : b.y)
    );
}
vec2 pick(in bvec2 cond, in float a, in vec2 b) 
{ 
    return vec2(
        (cond.x ? a : b.x),
        (cond.y ? a : b.y)
    );
}
vec2 pick(in bvec2 cond, in vec2 a, in float b) 
{ 
    return vec2(
        (cond.x ? a.x : b),
        (cond.y ? a.y : b)
    );
}

vec3 pick(in bvec3 cond, in vec3 a, in vec3 b) 
{ 
    return vec3(
        (cond.x ? a.x : b.x),
        (cond.y ? a.y : b.y),
        (cond.z ? a.z : b.z)
    );
}
vec3 pick(in bvec3 cond, in float a, in vec3 b) 
{ 
    return vec3(
        (cond.x ? a : b.x),
        (cond.y ? a : b.y),
        (cond.z ? a : b.z)
    );
}
vec3 pick(in bvec3 cond, in vec3 a, in float b) 
{ 
    return vec3(
        (cond.x ? a.x : b),
        (cond.y ? a.y : b),
        (cond.z ? a.z : b)
    );
}

vec4 pick(in bvec4 cond, in vec4 a, in vec4 b) 
{ 
    return vec4(
        (cond.x ? a.x : b.x),
        (cond.y ? a.y : b.y),
        (cond.z ? a.z : b.z),
        (cond.w ? a.w : b.w)
    );
}
vec4 pick(in bvec4 cond, in float a, in vec4 b) 
{ 
    return vec4(
        (cond.x ? a : b.x),
        (cond.y ? a : b.y),
        (cond.z ? a : b.z),
        (cond.w ? a : b.w)
    );
}
vec4 pick(in bvec4 cond, in vec4 a, in float b) 
{ 
    return vec4(
        (cond.x ? a.x : b),
        (cond.y ? a.y : b),
        (cond.z ? a.z : b),
        (cond.w ? a.w : b)
    );
}

int pick(in bool cond, in int a, in int b) 
{ 
    return (cond ? a : b); 
}

ivec2 pick(in bvec2 cond, in ivec2 a, in ivec2 b) 
{ 
    return ivec2(
        (cond.x ? a.x : b.x),
        (cond.y ? a.y : b.y)
    );
}

ivec3 pick(in bvec3 cond, in ivec3 a, in ivec3 b) 
{ 
    return ivec3(
        (cond.x ? a.x : b.x),
        (cond.y ? a.y : b.y),
        (cond.z ? a.z : b.z)
    );
}

ivec4 pick(in bvec4 cond, in ivec4 a, in ivec4 b) 
{ 
    return ivec4(
        (cond.x ? a.x : b.x),
        (cond.y ? a.y : b.y),
        (cond.z ? a.z : b.z),
        (cond.w ? a.w : b.w)
    );
}

#endif
