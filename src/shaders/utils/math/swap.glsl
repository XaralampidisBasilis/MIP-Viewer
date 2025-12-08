#ifndef SWAP
#define SWAP

void swap(inout float a, inout float b) { float t = a; a = b; b = t; }
void swap(inout vec2  a, inout vec2  b) { vec2  t = a; a = b; b = t; }
void swap(inout vec3  a, inout vec3  b) { vec3  t = a; a = b; b = t; }
void swap(inout vec4  a, inout vec4  b) { vec4  t = a; a = b; b = t; }

#endif // SWAP