#ifndef SMOOTH_SIGN
#define SMOOTH_SIGN

float smooth_sign(in float e, float v) { return smoothstep(-e, e, v) * 2.0 - 1.0; }
vec2  smooth_sign(in float e, vec2  v) { return smoothstep(-e, e, v) * 2.0 - 1.0; }
vec3  smooth_sign(in float e, vec3  v) { return smoothstep(-e, e, v) * 2.0 - 1.0; }
vec4  smooth_sign(in float e, vec4  v) { return smoothstep(-e, e, v) * 2.0 - 1.0; }

#endif // SMOOTH_SIGN