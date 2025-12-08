#ifndef PICK
#define PICK

float pick(in float a, in float b, in bool cond) 
{ 
    return (cond ? a : b); 
}

vec2 pick(in float a, in float b, in bvec2 cond) 
{ 
    return vec2(
        (cond.x ? b : a),
        (cond.y ? b : a)
    );
}

vec3 pick(in float a, in float b, in bvec3 cond) 
{ 
    return vec3(
        (cond.x ? b : a),
        (cond.y ? b : a),
        (cond.z ? b : a)
    );
}

vec4 pick(in float a, in float b, in bvec4 cond) 
{ 
    return vec4(
        (cond.x ? b : a),
        (cond.y ? b : a),
        (cond.z ? b : a),
        (cond.w ? b : a)
    );
}

vec2 pick(in vec2 a, in vec2 b, in bvec2 cond) 
{ 
    return vec2(
        (cond.x ? b.x : a.x),
        (cond.y ? b.y : a.y)
    );
}

vec3 pick(in vec3 a, in vec3 b, in bvec3 cond) 
{ 
    return vec3(
        (cond.x ? b.x : a.x),
        (cond.y ? b.y : a.y),
        (cond.z ? b.z : a.z)
    );
}

vec4 pick(in vec4 a, in vec4 b, in bvec4 cond) 
{ 
    return vec4(
        (cond.x ? b.x : a.x),
        (cond.y ? b.y : a.y),
        (cond.z ? b.z : a.z),
        (cond.w ? b.w : a.w)
    );
}

int pick(in int a, in int b, in bool cond) 
{ 
    return (cond ? b : a); 
}


ivec2 pick(in int a, in int b, in bvec2 cond) 
{ 
    return ivec2(
        (cond.x ? b : a),
        (cond.y ? b : a)
    );
}

ivec3 pick(in int a, in int b, in bvec3 cond) 
{ 
    return ivec3(
        (cond.x ? b : a),
        (cond.y ? b : a),
        (cond.z ? b : a)
    );
}

ivec4 pick(in int a, in int b, in bvec4 cond) 
{ 
    return ivec4(
        (cond.x ? b : a),
        (cond.y ? b : a),
        (cond.z ? b : a),
        (cond.w ? b : a)
    );
}

ivec2 pick(in ivec2 a, in ivec2 b, in bvec2 cond) 
{ 
    return ivec2(
        (cond.x ? b.x : a.x),
        (cond.y ? b.y : a.y)
    );
}

ivec3 pick(in ivec3 a, in ivec3 b, in bvec3 cond) 
{ 
    return ivec3(
        (cond.x ? b.x : a.x),
        (cond.y ? b.y : a.y),
        (cond.z ? b.z : a.z)
    );
}

ivec4 pick(in ivec4 a, in ivec4 b, in bvec4 cond) 
{ 
    return ivec4(
        (cond.x ? b.x : a.x),
        (cond.y ? b.y : a.y),
        (cond.z ? b.z : a.z),
        (cond.w ? b.w : a.w)
    );
}

#endif
