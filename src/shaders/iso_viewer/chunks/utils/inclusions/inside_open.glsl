#ifndef INSIDE_OPEN
#define INSIDE_OPEN

#ifndef AND
#include "../logical/and"
#endif

bool   inside_open(in float x, in float a, in float b) { return and(x > a, x < b); }
bvec2  inside_open(in vec2  x, in vec2  a, in vec2  b) { return and(greaterThan(x, a), lessThan(x, b)); }
bvec3  inside_open(in vec3  x, in vec3  a, in vec3  b) { return and(greaterThan(x, a), lessThan(x, b)); }
bvec4  inside_open(in vec4  x, in vec4  a, in vec4  b) { return and(greaterThan(x, a), lessThan(x, b)); }
bvec2  inside_open(in vec2  x, in float a, in float b) { return and(greaterThan(x, vec2(a)), lessThan(x, vec2(b))); }
bvec3  inside_open(in vec3  x, in float a, in float b) { return and(greaterThan(x, vec3(a)), lessThan(x, vec3(b))); }
bvec4  inside_open(in vec4  x, in float a, in float b) { return and(greaterThan(x, vec4(a)), lessThan(x, vec4(b))); }

#endif 
