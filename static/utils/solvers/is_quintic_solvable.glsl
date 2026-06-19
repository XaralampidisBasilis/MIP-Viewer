#ifndef IS_QUINTIC_SOLVABLE
#define IS_QUINTIC_SOLVABLE

#ifndef QUARTIC_ROOTS
#include "./quartic_roots"
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif
#ifndef SIGN_CHANGE
#include "../math/sign_change"
#endif

// compute if quintic polynomial c0 + c1x + c2x^2 + c3x^3 + c4x^4 + c5x^5 = 0 is solvable for x in [xa, xb]

bool is_quintic_solvable(in float c[6], in vec2 xa_xb)
{
    // compute quintic derivative coefficients
    float d[5] = float[5](
        c[1], 
        c[2] * 2.0, 
        c[3] * 3.0, 
        c[4] * 4.0, 
        c[5] * 5.0
    );

    // solve for the critical points of the quintic polynomial
    vec4 x0_x1_x2_x3 = quartic_roots(d);
    x0_x1_x2_x3 = clamp(x0_x1_x2_x3, xa_xb.x, xa_xb.y);

    // compute the quintic extrema values at the critical points
    vec4 y0_y1_y2_y3 = eval_poly(c, x0_x1_x2_x3);

    // compute the quintic boundary values
    vec2 ya_yb = eval_poly(c, xa_xb);

    // combine function values
    vec4 ya_y0_y1_y2 = vec4(ya_yb.x, y0_y1_y2_y3.xyz);
    vec3 y2_y3_yb = vec3(y0_y1_y2_y3.zw, ya_yb.y);

    // detect any sign change
    return sign_change(ya_y0_y1_y2) || sign_change(y2_y3_yb);
}

bool is_quintic_solvable(in float c[6], in vec2 xa_xb, in vec2 ya_yb)
{
    // compute quintic derivative coefficients
    float d[5] = float[5](
        c[1], 
        c[2] * 2.0, 
        c[3] * 3.0, 
        c[4] * 4.0, 
        c[5] * 5.0
    );

    // solve for the critical points of the quintic polynomial
    vec4 x0_x1_x2_x3 = quartic_roots(d);
    x0_x1_x2_x3 = clamp(x0_x1_x2_x3, xa_xb.x, xa_xb.y);

    // compute the quintic extrema values at the critical points
    vec4 y0_y1_y2_y3 = eval_poly(c, x0_x1_x2_x3);

    // combine function values
    vec4 ya_y0_y1_y2 = vec4(ya_yb.x, y0_y1_y2_y3.xyz);
    vec3 y2_y3_yb = vec3(y0_y1_y2_y3.zw, ya_yb.y);

    // detect any sign change
    return sign_change(ya_y0_y1_y2) || sign_change(y2_y3_yb);
}

#endif
