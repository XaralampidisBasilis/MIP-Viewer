#ifndef SUM
#define SUM

// Computes sum of float vectors

float sum(in vec2 v) { return dot(v, vec2(1.0)); }
float sum(in vec3 v) { return dot(v, vec3(1.0)); }
float sum(in vec4 v) { return dot(v, vec4(1.0)); }

#endif 
