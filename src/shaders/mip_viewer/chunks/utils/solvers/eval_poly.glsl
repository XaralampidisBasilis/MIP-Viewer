#ifndef EVAL_POLY
#define EVAL_POLY

// Evaluate polynomial and derivatives using Horner's method
// Coefficients are provided in ascending order:
//  p(t) = c0 + c1 t + c2 t^2 + ... + cn t^n


// quadratic
float eval_poly(in vec3 c, in float t) 
{
    float a1 = c.y + c.z * t; // c1 + c2*t
    float f = c.x + a1 * t; // c0 + (c1 + c2*t) * t = c0 + c1*t + c2*t^2
    return f;
}
vec2 eval_poly(in vec3 c, in vec2 t) 
{
    vec2 a1 = c.y + c.z * t;
    vec2 f = c.x + a1 * t;
    return f;
}
vec3 eval_poly(in vec3 c, in vec3 t) 
{
    vec3 a1 = c.y + c.z * t;
    vec3 f = c.x + a1 * t;
    return f;
}
vec4 eval_poly(in vec3 c, in vec4 t) 
{
    vec4 a1 = c.y + c.z * t;
    vec4 f = c.x + a1 * t;
    return f;
}

// cubic
float eval_poly(in vec4 c, in float t) 
{
    float a2 = c.z + c.w * t; // c2 + c3*t
    float a1 = c.y + a2 * t;  // c1 + (c2 + c3*t) * t = c1 + c2*t + c3*t^2 
    float f = c.x + a1 * t;         // c0 + (c1 + c2*t + c3*t^2) * t = c0 + c1*t + c2*t^2 + c3*t^3
    return f;
}
vec2 eval_poly(in vec4 c, in vec2 t) 
{
    vec2 a2 = c.z + c.w * t;
    vec2 a1 = c.y + a2 * t;
    vec2 f = c.x + a1 * t;
    return f;
}
vec3 eval_poly(in vec4 c, in vec3 t) 
{
    vec3 a2 = c.z + c.w * t;
    vec3 a1 = c.y + a2 * t;
    vec3 f = c.x + a1 * t;
    return f;
}
vec4 eval_poly(in vec4 c, in vec4 t) 
{
    vec4 a2 = c.z + c.w * t;
    vec4 a1 = c.y + a2 * t;
    vec4 f = c.x + a1 * t;
    return f;
}
float eval_poly(in vec4 c, in float t, out float f1) 
{
    float a2 = c.z + c.w * t; // c2 + c3*t
    float a1 = c.y + a2 * t;  // c1 + (c2 + c3*t) * t = c1 + c2*t + c3*t^2 
    float f = c.x + a1 * t;         // c0 + (c1 + c2*t + c3*t^2) * t = c0 + c1*t + c2*t^2 + c3*t^3

    float b2 = a2 + c.w * t;  // (c2 + c3*t) + c3*t = c2 + 2*c3*t
    f1 = a1 + b2 * t;         // (c1 + c2*t + c3*t^2) + (c2 + 2*c3*t) * t = c1 + 2*c2*t + 3*c3*t^2
    return f;
}
vec2 eval_poly(in vec4 c, in vec2 t, out vec2 f1) 
{
    vec2 a2 = c.z + c.w * t;
    vec2 a1 = c.y + a2 * t;
    vec2 f = c.x + a1 * t;

    vec2 b2 = a2 + c.w * t;
    f1 = a1 + b2 * t;
    return f;
}
vec3 eval_poly(in vec4 c, in vec3 t, out vec3 f1) 
{
    vec3 a2 = c.z + c.w * t;
    vec3 a1 = c.y + a2 * t;
    vec3 f = c.x + a1 * t;

    vec3 b2 = a2 + c.w * t;
    f1 = a1 + b2 * t;
    return f;
}
vec4 eval_poly(in vec4 c, in vec4 t, out vec4 f1) 
{
    vec4 a2 = c.z + c.w * t;
    vec4 a1 = c.y + a2 * t;
    vec4 f = c.x + a1 * t;

    vec4 b2 = a2 + c.w * t;
    f1 = a1 + b2 * t;
    return f;
}

#endif
