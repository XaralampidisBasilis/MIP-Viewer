#ifndef OR
#define OR

#ifndef ON
#include "./on"
#endif

bool  or(in bool  a, in bool  b) { return (a || b); }
bvec2 or(in bvec2 a, in bvec2 b) { return bvec2(a.x || b.x, a.y || b.y); }
bvec3 or(in bvec3 a, in bvec3 b) { return bvec3(a.x || b.x, a.y || b.y, a.z || b.z); }
bvec4 or(in bvec4 a, in bvec4 b) { return bvec4(a.x || b.x, a.y || b.y, a.z || b.z, a.w || b.w); }

bool  or(in int   a, in int   b) { return or(on(a), on(b)); }
bvec2 or(in ivec2 a, in ivec2 b) { return or(on(a), on(b)); }
bvec3 or(in ivec3 a, in ivec3 b) { return or(on(a), on(b)); }
bvec4 or(in ivec4 a, in ivec4 b) { return or(on(a), on(b)); }

bool  or(in float a, in float b) { return or(on(a), on(b)); }
bvec2 or(in vec2  a, in vec2  b) { return or(on(a), on(b)); }
bvec3 or(in vec3  a, in vec3  b) { return or(on(a), on(b)); }
bvec4 or(in vec4  a, in vec4  b) { return or(on(a), on(b)); }

#endif