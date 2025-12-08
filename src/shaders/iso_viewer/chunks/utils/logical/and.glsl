#ifndef AND
#define AND

bool  and(in bool  a, in bool  b) { return (a && b); }
bvec2 and(in bvec2 a, in bvec2 b) { return bvec2(a.x && b.x, a.y && b.y); }
bvec3 and(in bvec3 a, in bvec3 b) { return bvec3(a.x && b.x, a.y && b.y, a.z && b.z); }
bvec4 and(in bvec4 a, in bvec4 b) { return bvec4(a.x && b.x, a.y && b.y, a.z && b.z, a.w && b.w); }

#endif