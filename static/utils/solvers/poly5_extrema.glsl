
/* Soures
Finding Real Polynomial Roots on GPUs (https://momentsingraphics.de/GPUPolynomialRoots.html),
Shadertoy Spherical harmonics glyphs (https://www.shadertoy.com/view/dlGSDV),
Ray Tracing Spherical Harmonics Glyphs (https://momentsingraphics.de/VMV2023.html),
High-Performance Polynomial Solver Cem Yuksel (https://www.cemyuksel.com/research/polynomials/),
cyPolynomial.h class (https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h),
*/

#ifndef POLY5_EXTREMA
#define POLY5_EXTREMA

#ifndef POLY5_ROOTS
#include "../solvers/poly5_roots"
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

vec2 poly5_extrema(
    float poly5[6], 
    float begin, 
    float end
){
    float derivative[6];
    derivative[0] = poly5[1];
    derivative[1] = poly5[2] * 2.0;
    derivative[2] = poly5[3] * 3.0;
    derivative[3] = poly5[4] * 4.0;
    derivative[4] = poly5[5] * 5.0;
    derivative[5] = 0.0;
    
    float roots[6]; poly5_roots(roots, derivative, begin, end);
    roots[0] = clamp(roots[0], begin, end);
    roots[1] = clamp(roots[1], begin, end);
    roots[2] = clamp(roots[2], begin, end);
    roots[3] = clamp(roots[3], begin, end);
    roots[4] = clamp(roots[4], begin, end);
    roots[5] = begin;

    float extrema[6]; 
    extrema[0] = eval_poly(poly5, roots[0]);
    extrema[1] = eval_poly(poly5, roots[1]);
    extrema[2] = eval_poly(poly5, roots[2]);
    extrema[3] = eval_poly(poly5, roots[3]);
    extrema[4] = eval_poly(poly5, roots[4]);
    extrema[5] = eval_poly(poly5, roots[5]);

    return vec2(mmin(extrema), mmax(extrema));
}

#endif