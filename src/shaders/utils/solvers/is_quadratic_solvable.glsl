#ifndef IS_QUADRATIC_SOLVABLE
#define IS_QUADRATIC_SOLVABLE

#ifndef LINEAR_ROOTS
#include "./linear_root"
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif
#ifndef SIGN_CHANGE
#include "../math/sign_change"
#endif

// compute if quadratic polynomial c0 + c1x + c2x^2 = y is solvable for x in [xa, xb]

bool is_quadratic_solvable(in vec3 c, in vec2 xa_xb)
{
    // compute quadratic derivative coefficients
    vec2 d = vec2(c.y, c.z * 2.0);

    // solve for the critical point of the quadratic polynomial
    float x0 = linear_root(d);
    x0 = clamp(x0, xa_xb.x, xa_xb.y);

    // compute the quadratic extrema value at the critical point
    float y0 = eval_poly(c, x0);

    // compute the quadratic at the boundaries
    vec2 ya_yb = eval_poly(c, xa_xb);

    // combine function values into a single vector
    vec3 ya_y0_yb = vec3(ya_yb.x, y0, ya_yb.y);

    // return result
    return sign_change(ya_y0_yb);
}

bool is_quadratic_solvable(in vec3 c, in vec2 xa_xb, in vec2 ya_yb)
{
    // compute quadratic derivative coefficients
    vec2 d = vec2(c.y, c.z * 2.0);

    // solve for the critical point of the quadratic polynomial
    float x0 = linear_root(d);
    x0 = clamp(x0, xa_xb.x, xa_xb.y);

    // compute the quadratic extrema value at the critical point
    float y0 = eval_poly(c, x0);

    // combine function values into a single vector
    vec3 ya_y0_yb = vec3(ya_yb.x, y0, ya_yb.y);

    // return result
    return sign_change(ya_y0_yb);
}

#endif
