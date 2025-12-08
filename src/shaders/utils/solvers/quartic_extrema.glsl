#ifndef QUARTIC_EXTREMA
#define QUARTIC_EXTREMA

#ifndef CUBIC_ROOTS
#include "./cubic_roots"
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif

// compute quartic extrema c0 + c1x + c2x^2 + c3x^3 + c4x^4 = 0

vec3 quartic_extrema(in float c[5])
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

    // compute the quartic extrema values at the critical points
    vec3 y0_y1_y2 = eval_poly(c, x0_x1_x2);

    // return result
    return y0_y1_y2;
}

// compute quartic extrema c0 + c1x + c2x^2 + c3x^3 + c4x^4 for x in [xa, xb]

bool quartic_extrema(in float c[5], in vec2 xa_xb)
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

    // return result
    return y0_y1_y2;
}

#endif
