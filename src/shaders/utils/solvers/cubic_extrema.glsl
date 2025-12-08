/* Sources
High-Performance Polynomial Solver Cem Yuksel (https://www.cemyuksel.com/research/polynomials/)
cyPolynomial.h class (https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h)
*/

#ifndef CUBIC_EXTREMA
#define CUBIC_EXTREMA

#ifndef QUADRATIC_ROOTS
#include "./quadratic_roots"
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif

// compute cubic extrema c0 + c1x + c2x^2 + c3x^3 

vec2 cubic_extrema(in vec4 c)
{
    // compute cubic derivative coefficients
    vec3 d = vec3(c.y, c.z * 2.0, c.w * 3.0);

    // solve for the critical points of the cubic polynomial
    vec2 x0_x1 = quadratic_roots(d);

    // compute the cubic extrema values at the critical points
    vec2 y0_y1 = eval_poly(c, x0_x1);

    return y0_y1;
}

// compute cubic extrema c0 + c1x + c2x^2 + c3x^3 for x in [xa, xb]

vec2 cubic_extrema(in vec4 c, in vec2 xa_xb)
{
    // compute cubic derivative coefficients
    vec3 d = vec3(c.y, c.z * 2.0, c.w * 3.0);;

    // solve for the critical points of the cubic polynomial
    vec2 x0_x1 = quadratic_roots(d);
    x0_x1 = clamp(x0_x1, xa_xb.x, xa_xb.y);

    // compute the cubic extrema values at the critical points
    vec2 y0_y1 = eval_poly(c, x0_x1);

    // return result
    return y0_y1;
}

#endif

