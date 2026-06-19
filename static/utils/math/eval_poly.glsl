#ifndef EVAL_POLY
#define EVAL_POLY

#ifndef EVAL_POLY_MAX_DEGREE
#define EVAL_POLY_MAX_DEGREE 6
#endif

#ifndef EVAL_POLY_MAX_DERIVATIVE
#define EVAL_POLY_MAX_DERIVATIVE 5
#endif

// Evaluate polynomial and derivatives using Horner's method
// Coefficients are provided in ascending order:
//  p(t) = c0 + c1 t + c2 t^2 + ... + cn t^n

// linear 
float eval_poly(in vec2 c, in float t) 
{
    float f = c.x + c.y * t;        // c0 + c1*t
    return f;
}
vec2 eval_poly(in vec2 c, in vec2 t) 
{
    vec2 f = c.x + c.y * t;
    return f;
}
vec3 eval_poly(in vec2 c, in vec3 t) 
{
    vec3 f = c.x + c.y * t;
    return f;
}
vec4 eval_poly(in vec2 c, in vec4 t) 
{
    vec4 f = c.x + c.y * t;
    return f;
}

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

float eval_poly(in vec3 c, in float t, out float f1) 
{
    float a1 = c.y + c.z * t; // c1 + c2*t
    float f = c.x + a1 * t;        // c0 + (c1 + c2*t) * t = c0 + c1*t + c2*t^2
    f1 = a1 + c.z * t;        // (c1 + c2*t) + c2*t = c1 + 2*c2*t
    return f;
}
vec2 eval_poly(in vec3 c, in vec2 t, out vec2 f1) 
{
    vec2 a1 = c.y + c.z * t;
    vec2 f = c.x + a1 * t;
    f1 = a1 + c.z * t;
    return f;
}
vec3 eval_poly(in vec3 c, in vec3 t, out vec3 f1) 
{
    vec3 a1 = c.y + c.z * t;
    vec3 f = c.x + a1 * t;
    f1 = a1 + c.z * t;
    return f;
}
vec4 eval_poly(in vec3 c, in vec4 t, out vec4 f1) 
{
    vec4 a1 = c.y + c.z * t;
    vec4 f = c.x + a1 * t;
    f1 = a1 + c.z * t;
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
float eval_poly(in vec4 c, in float t, out float f1, out float f2) 
{
    float a2 = c.z + c.w * t; // c2 + c3*t
    float a1 = c.y + a2 * t;  // c1 + (c2 + c3*t) * t = c1 + c2*t + c3*t^2 
    float f = c.x + a1 * t;         // c0 + (c1 + c2*t + c3*t^2) * t = c0 + c1*t + c2*t^2 + c3*t^3

    float b2 = a2 + c.w * t;  // (c2 + c3*t) + c3*t = c2 + 2*c3*t
    f1 = a1 + b2 * t;         // (c1 + c2*t + c3*t^2) + (c2 + 2*c3*t) * t = c1 + 2*c2*t + 3*c3*t^2
    
    float b1 = b2 + c.w * t;  // (c2 + 2*c3*t) + c3*t = c2 + 3*c3*t
    f2 = b1 * 2.0;            // (c2 + 3*c3*t) * 2.0 = 2*c2 + 6*c3*t
    return f;
}
vec2 eval_poly(in vec4 c, in vec2 t, out vec2 f1, out vec2 f2) 
{
    vec2 a2 = c.z + c.w * t;
    vec2 a1 = c.y + a2 * t;
    vec2 f = c.x + a1 * t;

    vec2 b2 = a2 + c.w * t;
    f1 = a1 + b2 * t;

    vec2 b1 = b2 + c.w * t;
    f2 = b1 * 2.0;
    return f;
}
vec3 eval_poly(in vec4 c, in vec3 t, out vec3 f1, out vec3 f2) 
{
    vec3 a2 = c.z + c.w * t;
    vec3 a1 = c.y + a2 * t;
    vec3 f = c.x + a1 * t;

    vec3 b2 = a2 + c.w * t;
    f1 = a1 + b2 * t;

    vec3 b1 = b2 + c.w * t;
    f2 = b1 * 2.0;
    return f;
}
vec4 eval_poly(in vec4 c, in vec4 t, out vec4 f1, out vec4 f2) 
{
    vec4 a2 = c.z + c.w * t;
    vec4 a1 = c.y + a2 * t;
    vec4 f = c.x + a1 * t;

    vec4 b2 = a2 + c.w * t;
    f1 = a1 + b2 * t;

    vec4 b1 = b2 + c.w * t;
    f2 = b1 * 2.0;
    return f;
}

// quartic
float eval_poly(in float c[5], in float t) 
{
    float f = c[4];
    f = f * t + c[3];
    f = f * t + c[2];
    f = f * t + c[1];
    f = f * t + c[0];
    return f;
}
vec2 eval_poly(in float c[5], in vec2 t) 
{
    vec2 f = vec2(c[4]);
    f = f * t + c[3];
    f = f * t + c[2];
    f = f * t + c[1];
    f = f * t + c[0];
    return f;
}
vec3 eval_poly(in float c[5], in vec3 t) 
{
    vec3 f = vec3(c[4]);
    f = f * t + c[3];
    f = f * t + c[2];
    f = f * t + c[1];
    f = f * t + c[0];
    return f;
}
vec4 eval_poly(in float c[5], in vec4 t) 
{
    vec4 f = vec4(c[4]);
    f = f * t + c[3];
    f = f * t + c[2];
    f = f * t + c[1];
    f = f * t + c[0];
    return f;
}
float eval_poly(in float c[5], in float t, out float f1) 
{
    float f = c[4];
    f1 = f;
    f  = f * t + c[3];
    f1 = f1 * t + f;
    f  = f * t + c[2];
    f1 = f1 * t + f;
    f  = f * t + c[1];
    f1 = f1 * t + f;
    f  = f * t + c[0];
    return f;
}
vec2 eval_poly(in float c[5], in vec2 t, out vec2 f1) 
{
    vec2 f = vec2(c[4]);
    f1 = f;
    f  = f * t + c[3];
    f1 = f1 * t + f;
    f  = f * t + c[2];
    f1 = f1 * t + f;
    f  = f * t + c[1];
    f1 = f1 * t + f;
    f  = f * t + c[0];
    return f;
}
vec3 eval_poly(in float c[5], in vec3 t, out vec3 f1) 
{
    vec3 f = vec3(c[4]);
    f1 = f;
    f  = f * t + c[3];
    f1 = f1 * t + f;
    f  = f * t + c[2];
    f1 = f1 * t + f;
    f  = f * t + c[1];
    f1 = f1 * t + f;
    f  = f * t + c[0];
    return f;
}
vec4 eval_poly(in float c[5], in vec4 t, out vec4 f1) 
{
    vec4 f = vec4(c[4]);
    f1 = f;
    f  = f * t + c[3];
    f1 = f1 * t + f;
    f  = f * t + c[2];
    f1 = f1 * t + f;
    f  = f * t + c[1];
    f1 = f1 * t + f;
    f  = f * t + c[0];
    return f;
}

// quintic 
float eval_poly(in float c[6], in float t) 
{
    float f = c[5];
    f = f * t + c[4];
    f = f * t + c[3];
    f = f * t + c[2];
    f = f * t + c[1];
    f = f * t + c[0];
    return f;
}
vec2 eval_poly(in float c[6], in vec2 t) 
{
    vec2 f = vec2(c[5]);
    f = f * t + c[4];
    f = f * t + c[3];
    f = f * t + c[2];
    f = f * t + c[1];
    f = f * t + c[0];
    return f;
}
vec3 eval_poly(in float c[6], in vec3 t) 
{
    vec3 f = vec3(c[5]);
    f = f * t + c[4];
    f = f * t + c[3];
    f = f * t + c[2];
    f = f * t + c[1];
    f = f * t + c[0];
    return f;
}
vec4 eval_poly(in float c[6], in vec4 t) 
{
    vec4 f = vec4(c[5]);
    f = f * t + c[4];
    f = f * t + c[3];
    f = f * t + c[2];
    f = f * t + c[1];
    f = f * t + c[0];
    return f;
}
float eval_poly(in float c[6], in float t, out float f1) 
{
    float f  = c[5];
    f1 = f;
    f  = f * t + c[4];
    f1 = f1 * t + f;
    f  = f * t + c[3];
    f1 = f1 * t + f;
    f  = f * t + c[2];
    f1 = f1 * t + f;
    f  = f * t + c[1];
    f1 = f1 * t + f;
    f  = f * t + c[0];
    return f;
}
vec2 eval_poly(in float c[6], in vec2 t, out vec2 f1) 
{
    vec2 f  = vec2(c[5]);
    f1 = f;
    f  = f * t + c[4];
    f1 = f1 * t + f;
    f  = f * t + c[3];
    f1 = f1 * t + f;
    f  = f * t + c[2];
    f1 = f1 * t + f;
    f  = f * t + c[1];
    f1 = f1 * t + f;
    f  = f * t + c[0];
    return f;
}
vec3 eval_poly(in float c[6], in vec3 t, out vec3 f1) 
{
    vec3 f = vec3(c[5]);
    f1 = f;
    f  = f * t + c[4];
    f1 = f1 * t + f;
    f  = f * t + c[3];
    f1 = f1 * t + f;
    f  = f * t + c[2];
    f1 = f1 * t + f;
    f  = f * t + c[1];
    f1 = f1 * t + f;
    f  = f * t + c[0];
    return f;
}
vec4 eval_poly(in float c[6], in vec4 t, out vec4 f1) 
{
    vec4 f = vec4(c[5]);
    f1 = f;
    f  = f * t + c[4];
    f1 = f1 * t + f;
    f  = f * t + c[3];
    f1 = f1 * t + f;
    f  = f * t + c[2];
    f1 = f1 * t + f;
    f  = f * t + c[1];
    f1 = f1 * t + f;
    f  = f * t + c[0];
    return f;
}

// general polynomial

// float eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in float t) 
// {    
//     float f = c[EVAL_POLY_MAX_DEGREE];

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f = f * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
    
//     return f;
// }
// vec2 eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in vec2 t) 
// {    
//     vec2 f = vec2(c[EVAL_POLY_MAX_DEGREE]);

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f = f * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
    
//     return f;
// }
// vec3 eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in vec3 t) 
// {    
//     vec3 f = vec3(c[EVAL_POLY_MAX_DEGREE]);

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f = f * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
    
//     return f;
// }
// vec4 eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in vec4 t) 
// {    
//     vec4 f = vec4(c[EVAL_POLY_MAX_DEGREE]);

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f = f * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
    
//     return f;
// }
// float eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in float t) 
// {    
//     float f = c[EVAL_POLY_MAX_DEGREE];

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f = f * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
//     return f;
// }
// vec2 eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in vec2 t) 
// {    
//     vec2 f = vec2(c[EVAL_POLY_MAX_DEGREE]);

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f = f * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
//     return f;
// }
// vec3 eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in vec3 t) 
// {    
//     vec3 f = vec3(c[EVAL_POLY_MAX_DEGREE]);

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f = f * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
//     return f;
// }
// vec4 eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in vec4 t) 
// {    
//     vec4 f = vec4(c[EVAL_POLY_MAX_DEGREE]);

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f = f * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
//     return f;
// }
// void eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in float t, out float f, out float f1)
// {
//     f1 = 0.0;
//     f = c[EVAL_POLY_MAX_DEGREE];

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f1 = f1 * t + f;
//         f = f * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
// }
// void eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in vec2 t, out vec2 f, out vec2 f1)
// {
//     f1 = vec2(0.0);
//     f = vec2(c[EVAL_POLY_MAX_DEGREE]);

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f1 = f1 * t + f;
//         f  = f  * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
// }
// void eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in vec3 t, out vec3 f, out vec3 f1)
// {
//     f1 = vec3(0.0);
//     f = vec3(c[EVAL_POLY_MAX_DEGREE]);

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f1 = f1 * t + f;
//         f  = f  * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
// }
// void eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in vec4 t, out vec4 f, out vec4 f1)
// {
//     f1 = vec4(0.0);
//     f = vec4(c[EVAL_POLY_MAX_DEGREE]);

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f1 = f1 * t + f;
//         f  = f  * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
// }
// void eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in float t, out float f, out float f1, out float f2)
// {
//     f2 = 0.0; 
//     f1 = 0.0;
//     f = c[EVAL_POLY_MAX_DEGREE];

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f2 = f2 * t + 2.0 * f1;
//         f1 = f1 * t + f;
//         f = f * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
// }
// void eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in vec2 t, out vec2 f, out vec2 f1, out vec2 f2)
// {
//     f2 = vec2(0.0); 
//     f1 = vec2(0.0);
//     f  = vec2(c[EVAL_POLY_MAX_DEGREE]);

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f2 = f2 * t + 2.0 * f1;
//         f1 = f1 * t + f;
//         f  = f  * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
// }
// void eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in vec3 t, out vec3 f, out vec3 f1, out vec3 f2)
// {
//     f2 = vec3(0.0); 
//     f1 = vec3(0.0);
//     f  = vec3(c[EVAL_POLY_MAX_DEGREE]);

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f2 = f2 * t + 2.0 * f1;
//         f1 = f1 * t + f;
//         f  = f  * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
// }
// void eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in vec4 t, out vec4 f, out vec4 f1, out vec4 f2)
// {
//     f2 = vec4(0.0); 
//     f1 = vec4(0.0);
//     f  = vec4(c[EVAL_POLY_MAX_DEGREE]);

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         f2 = f2 * t + 2.0 * f1;
//         f1 = f1 * t + f;
//         f  = f  * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
// }
// void eval_poly(in float c[EVAL_POLY_MAX_DEGREE + 1], in float t, out float D[EVAL_POLY_MAX_DERIVATIVE + 1])  // D[0] = f, D[1] = f1, ..., D[d] = fd
// {
//     for (int d = EVAL_POLY_MAX_DERIVATIVE; d >= 1; --d) 
//     {
//         D[d] = 0.0;
//     }
//     D[0] = c[EVAL_POLY_MAX_DEGREE];

//     #pragma unroll
//     for (int i = 0; i < EVAL_POLY_MAX_DEGREE; i++) 
//     {
//         for (int d = EVAL_POLY_MAX_DERIVATIVE; d >= 1; --d) 
//         {
//             D[d] = D[d] * t + float(d) * D[d - 1];
//         }
//         D[0] = D[0]  * t + c[EVAL_POLY_MAX_DEGREE - 1 - i];
//     }
// }

#endif
