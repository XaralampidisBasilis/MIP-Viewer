#ifndef MEAN
#define MEAN

// vec and ivec sum using dot product for compactness and efficiency
float mean(in vec2 v) { return dot(v, vec2(1.0)) / 2.0; }
float mean(in vec3 v) { return dot(v, vec3(1.0)) / 3.0; }
float mean(in vec4 v) { return dot(v, vec4(1.0)) / 4.0; }

float mean(in float a, in float b) { return (a + b) / 2.0; }
vec2  mean(in vec2  a, in vec2  b) { return (a + b) / 2.0; }
vec3  mean(in vec3  a, in vec3  b) { return (a + b) / 2.0; }
vec4  mean(in vec4  a, in vec4  b) { return (a + b) / 2.0; }

float mean(in float a, in float b, in float c) { return (a + b + c) / 3.0; }
vec2  mean(in vec2  a, in vec2  b, in vec2  c) { return (a + b + c) / 3.0; }
vec3  mean(in vec3  a, in vec3  b, in vec3  c) { return (a + b + c) / 3.0; }
vec4  mean(in vec4  a, in vec4  b, in vec4  c) { return (a + b + c) / 3.0; }

#endif // MEAN
