#ifndef INSIDE_CLOSED
#define INSIDE_CLOSED

#ifndef AND
#include "../logical/and"
#endif

bool   inside_closed(in float x, in float a, in float b) { return and(x >= a, x <= b); }
bvec2  inside_closed(in vec2  x, in vec2  a, in vec2  b) { return and(greaterThanEqual(x, a), lessThanEqual(x, b)); }
bvec3  inside_closed(in vec3  x, in vec3  a, in vec3  b) { return and(greaterThanEqual(x, a), lessThanEqual(x, b)); }
bvec4  inside_closed(in vec4  x, in vec4  a, in vec4  b) { return and(greaterThanEqual(x, a), lessThanEqual(x, b)); }
bvec2  inside_closed(in vec2  x, in float a, in float b) { return and(greaterThanEqual(x, vec2(a)), lessThanEqual(x, vec2(b))); }
bvec3  inside_closed(in vec3  x, in float a, in float b) { return and(greaterThanEqual(x, vec3(a)), lessThanEqual(x, vec3(b))); }
bvec4  inside_closed(in vec4  x, in float a, in float b) { return and(greaterThanEqual(x, vec4(a)), lessThanEqual(x, vec4(b))); }

#endif 