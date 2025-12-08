
/* Soures
Based on Blinn's paper (https://courses.cs.washington.edu/courses/cse590b/13au/lecture_notes/solvecubic_p5.pdf),
Article by Christoph Peters (https://momentsingraphics.de/CubicRoots.html#_Blinn07b),
Shadertoy Cubic Equation Solver II (https://www.shadertoy.com/view/7tBGzK),
Shadertoy Quartic Reflections https://www.shadertoy.com/view/flBfzm,
*/

#ifndef CUBIC_ROOTS
#define CUBIC_ROOTS

#ifndef QUADRATIC_ROOTS
#include "./quadratic_roots"
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif
#ifndef CBRT
#include "../math/cbrt"
#endif
#ifndef PICK
#include "../math/pick"
#endif
#ifndef SQRT_3
#define SQRT_3 1.73205080757
#endif
#ifndef NAN
#define NAN uintBitsToFloat(0x7fc00000u)
#endif

// Solves the cubic equation: c0 + c1*x^1 + c2*x^2 + c3x^3 = 0
// We assume non zero cubic coefficient
// x0 is the fallback root

vec3 cubic_roots(in vec4 c)
{
    // Flip to minimize instability
    bool flip = abs(c.z * c.x) >= abs(c.y * c.w);
    vec3 n = flip ? c.wzy / c.x : c.xyz / c.w;
    n.yz /= 3.0;

    // compute hessian coefficients eq(0.4)
    vec3 h = vec3(
        n.y - n.z * n.z,                          // δ1 = c.w * c.y - c.z^2
        n.x - n.y * n.z,                          // δ2 = c.w * c.x - c.y * c.z
        dot(vec2(n.z, -n.y), n.xy)    // δ3 = c.z * c.x - c.y^2
    );
    h.y /= 2.0;

    // compute cubic discriminant eq(0.7)
    float d = dot(vec2(h.x, -h.y), h.zy); // Δ = δ1 * δ3 - δ2^2
    float sqrt_d = sqrt(abs(d));

    // compute depressed cubic eq(0.16), r[0] + r[1] * x + x^3 eq(0.11) eq(0.16)
    vec2 r = vec2(h.y - h.x * n.z, h.x);
    
    // compute real root using cubic root formula for one real and two complex roots eq(0.15)
    vec3 x1 = vec3(cbrt(-r.x + sqrt_d) + cbrt(-r.x - sqrt_d));

    // compute cubic roots using complex number formula eq(0.14)  
    float t = atan(sqrt_d, -r.x) / 3.0;
    mat2 proj = mat2(-1.0, -SQRT_3, -1.0, SQRT_3);

    // compute three roots via rotation, applying complex root formula eq(0.14)
    vec2 x2 = vec2(cos(t), sin(t));
    vec3 x3 = vec3(x2 * proj, x2.x * 2.0);

    // revert transformation eq(0.2) and eq(0.16)
    x3 *= sqrt(max(0.0, -r.y)); 

    // choose cubic roots based on discriminant sign 
    vec3 x = (d > 0.0) ? x3 : x1;
    x = x - n.z;
    x = flip ? 1.0 / x : x;

    // Improve numerical stability of roots with Newton–Raphson correction
    vec3 y, dydx;
    y = eval_poly(c, x, dydx);
    x -= y / dydx; 
    y = eval_poly(c, x, dydx);
    x -= y / dydx; 

    // return result
    return x;
}

vec3 cubic_roots_2(in vec4 c)
{
    // Normalize
    vec3 n = c.xyz / c.w;
    n.yz /= 3.0;

    // compute hessian coefficients eq(0.4)
    vec3 h = vec3(
        n.y - n.z * n.z,                          // δ1 = c.w * c.y - c.z^2
        n.x - n.y * n.z,                          // δ2 = c.w * c.x - c.y * c.z
        dot(vec2(n.z, -n.y), n.xy)    // δ3 = c.z * c.x - c.y^2
    );
    h.y /= 2.0;

    // compute cubic discriminant eq(0.7)
    float d = dot(vec2(h.x, -h.y), h.zy); // Δ = δ1 * δ3 - δ2^2
    float sqrt_d = sqrt(abs(d));

    // compute depressed cubic eq(0.16), r[0] + r[1] * x + x^3 eq(0.11) eq(0.16)
    vec2 r = vec2(h.y - h.x * n.z, h.x);
    
    // compute real root using cubic root formula for one real and two complex roots eq(0.15)
    vec3 x1 = vec3(cbrt(-r.x + sqrt_d) + cbrt(-r.x - sqrt_d));

    // compute cubic roots using complex number formula eq(0.14)  
    float t = atan(sqrt_d, -r.x) / 3.0;
    mat2 proj = mat2(-1.0, -SQRT_3, -1.0, SQRT_3);

    // compute three roots via rotation, applying complex root formula eq(0.14)
    vec2 x2 = vec2(cos(t), sin(t));
    vec3 x3 = vec3(x2 * proj, x2.x * 2.0);

    // revert transformation eq(0.2) and eq(0.16)
    x3 *= sqrt(max(0.0, -r.y)); 

    // choose cubic roots based on discriminant sign 
    vec3 x = (d > 0.0) ? x3 : x1;
    x = x - n.z;

    // return result
    return x;
}

#endif





