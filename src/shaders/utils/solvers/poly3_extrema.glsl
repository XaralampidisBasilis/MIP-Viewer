
/* Soures
Finding Real Polynomial Roots on GPUs (https://momentsingraphics.de/GPUPolynomialRoots.html),
Shadertoy Spherical harmonics glyphs (https://www.shadertoy.com/view/dlGSDV),
Ray Tracing Spherical Harmonics Glyphs (https://momentsingraphics.de/VMV2023.html),
High-Performance Polynomial Solver Cem Yuksel (https://www.cemyuksel.com/research/polynomials/),
cyPolynomial.h class (https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h),
*/

#ifndef POLY3_EXTREMA
#define POLY3_EXTREMA

#ifndef POLY3_ROOTS
#include "../solvers/poly3_roots"
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif
#ifndef MMIN
#include "../math/mmin"
#endif
#ifndef MMAX
#include "../math/mmax"
#endif

vec2 poly3_extrema
(
    vec4 poly3, 
    float begin, 
    float end
){
    vec4 derivative = vec4(
        poly3[1],
        poly3[2] * 2.0,
        poly3[3] * 3.0,
        0.0
    );

    vec4 roots; poly3_roots(roots, derivative, begin, end);
    roots[0] = clamp(roots[0], begin, end);
    roots[1] = clamp(roots[1], begin, end);
    roots[2] = clamp(roots[2], begin, end);
    roots[3] = begin;

    vec4 extrema = eval_poly(poly3, roots);
    return vec2(mmin(extrema), mmax(extrema));
}

#endif