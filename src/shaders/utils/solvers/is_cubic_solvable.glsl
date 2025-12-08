/* Sources
High-Performance Polynomial Solver Cem Yuksel (https://www.cemyuksel.com/research/polynomials/)
cyPolynomial.h class (https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h)
*/

#ifndef IS_CUBIC_SOLVABLE
#define IS_CUBIC_SOLVABLE

#ifndef QUADRATIC_ROOTS
#include "./quadratic_roots"
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif
#ifndef SIGN_CHANGE
#include "../math/sign_change"
#endif

// compute if cubic polynomial c0 + c1x + c2x^2 + c3x^3 = y is solvable for x in [xa, xb]

bool is_cubic_solvable(in vec4 c, in vec2 xa_xb)
{
    // compute cubic derivative coefficients
    vec3 d = vec3(c.y, c.z * 2.0, c.w * 3.0);;

    // solve for the critical points of the cubic polynomial
    vec2 x0_x1 = quadratic_roots(d);
    x0_x1 = clamp(x0_x1, xa_xb.x, xa_xb.y);

    // compute the cubic extrema values at the critical points
    vec2 y0_y1 = eval_poly(c, x0_x1);

  // compute the cubic at the boundaries
    vec2 ya_yb = eval_poly(c, xa_xb);

    // combine function values
    vec4 ya_y0_y1_yb = vec4(ya_yb.x, y0_y1, ya_yb.y);

    // return result
    return sign_change(ya_y0_y1_yb);
}

bool is_cubic_solvable(in vec4 c, in vec2 xa_xb, in vec2 ya_yb)
{ 
    // compute cubic derivative coefficients
    vec3 d = vec3(c.y, c.z * 2.0, c.w * 3.0);;

    // solve for the critical points of the cubic polynomial
    vec2 x0_x1 = quadratic_roots(d);
    x0_x1 = clamp(x0_x1, xa_xb.x, xa_xb.y);

    // compute the cubic extrema values at the critical points
    vec2 y0_y1;
    y0_y1 = eval_poly(c, x0_x1);

     // combine function values
    vec4 ya_y0_y1_yb = vec4(ya_yb.x, y0_y1, ya_yb.y);

    // return result
    return sign_change(ya_y0_y1_yb);
}

#endif

