#ifndef QUINTIC_EXTREMA
#define QUINTIC_EXTREMA

#ifndef QUARTIC_ROOTS
#include "./quartic_roots"
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif

// compute quintic extrema c0 + c1x + c2x^2 + c3x^3 + c4x^4 + c5x^5 = 0

vec4 quintic_extrema(in float c[6])
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

    // compute the quintic extrema values at the critical points
    vec4 y0_y1_y2_y3 = eval_poly(c, x0_x1_x2_x3);

    // detect any sign change
    return y0_y1_y2_y3;
}

vec4 quintic_extrema(in float c[6], in vec2 xa_xb)
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

    // detect any sign change
    return y0_y1_y2_y3;
}


#endif
