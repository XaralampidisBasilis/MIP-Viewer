#ifndef IS_QUARTIC_SOLVABLE
#define IS_QUARTIC_SOLVABLE

#ifndef CUBIC_ROOTS
#include "./cubic_roots"
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif

// compute if quartic polynomial c0 + c1x + c2x^2 + c3x^3 + c4x^4 = y is solvable for x in [xa, xb]

bool is_quartic_solvable(in float c[5], in vec2 xa_xb)
{
    // compute quartic derivative coefficients
    vec4 d = vec4(
        c[1], 
        c[2] * 2.0, 
        c[3] * 3.0, 
        c[4] * 4.0
    );

    // solve for the critical points of the quartic polynomial
    vec3 x0_x1_x2 = cubic_roots(d);
    x0_x1_x2 = clamp(x0_x1_x2, xa_xb.x, xa_xb.y);

    // compute the quartic extrema values at the critical points
    vec3 y0_y1_y2 = eval_poly(c, x0_x1_x2);

    // compute the quartic at the boundaries
    vec2 ya_yb = eval_poly(c, xa_xb);

    // combine function values
    vec4 ya_y0_y1_y2 = vec4(ya_yb.x, y0_y1_y2);
    vec4 y0_y1_y2_yb = vec4(y0_y1_y2, ya_yb.y);

    // return result
    return sign_change(ya_y0_y1_y2) || sign_change(y0_y1_y2_yb);
}

bool is_quartic_solvable(in float c[5], in vec2 xa_xb, in vec2 ya_yb)
{
    // compute quartic derivative coefficients
    vec4 d = vec4(
        c[1], 
        c[2] * 2.0, 
        c[3] * 3.0, 
        c[4] * 4.0
    );

    // solve for the critical points of the quartic polynomial
    vec3 x0_x1_x2 = cubic_roots(d);
    x0_x1_x2 = clamp(x0_x1_x2, xa_xb.x, xa_xb.y);

    // compute the quartic extrema values at the critical points
    vec3 y0_y1_y2 = eval_poly(c, x0_x1_x2);

    // combine function values
    vec4 ya_y0_y1_y2 = vec4(ya_yb.x, y0_y1_y2);
    vec4 y0_y1_y2_yb = vec4(y0_y1_y2, ya_yb.y);

    // return result
    return sign_change(ya_y0_y1_y2) || sign_change(y0_y1_y2_yb);
}

#endif
