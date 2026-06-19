#ifndef SUM
#define SUM

// Computes sum of float vectors

float sum(in vec2 v) { return dot(v, vec2(1.0)); }
float sum(in vec3 v) { return dot(v, vec3(1.0)); }
float sum(in vec4 v) { return dot(v, vec4(1.0)); }

// Computes sum of float arrays up to length 9
float sum(in float v[2]) { return v[0] + v[1]; }
float sum(in float v[3]) { return v[0] + v[1] + v[2]; }
float sum(in float v[4]) { return v[0] + v[1] + v[2] + v[3]; }
float sum(in float v[5]) { return v[0] + v[1] + v[2] + v[3] + v[4]; }
float sum(in float v[6]) { return v[0] + v[1] + v[2] + v[3] + v[4] + v[5]; }
float sum(in float v[7]) { return v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6]; }
float sum(in float v[8]) { return v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7]; }
float sum(in float v[9]) { return v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7] + v[8]; }

#endif 
