/* Sources
A UNIVERSAL METHOD OF SOLVING QUARTIC EQUATIONS, International Journal of Pure and Applied Mathematics
(https://www.ijpam.eu/contents/2011-71-2/7/7.pdf),
Wikipedia quartic equation
(https://www.wikiwand.com/en/articles/Quartic_equation)
*/

#ifndef QUARTIC_ROOTS_NEW
#define QUARTIC_ROOTS_NEW

#ifndef PICK
#include "../math/pick"
#endif
#ifndef SSIGN
#include "../math/ssign"
#endif
#ifndef CBRT
#include "../math/cbrt"
#endif
#ifndef NAN
#define NAN uintBitsToFloat(0x7fc00000u)
#endif
#ifndef MICRO_TOLERANCE
#define MICRO_TOLERANCE 1e-6
#endif

// Solve resolvent depressed cubic 
// rc + rby + y^3 = 0
float resolvent_depressed_cubic_max_root(in float rc, in float rb)
{
    // normalize coefficients
    vec2 r = vec2(rc / 2.0, rb / 3.0);

    // compute cubic discriminant 
    float d = dot(vec2(1.0, r.y), r * r); 
    float sqrt_d = sqrt(abs(d));
    
    // compute real root using cubic root formula for one real and two complex roots eq(0.15)
    float y1 = 
        cbrt(-r.x + sqrt_d) + 
        cbrt(-r.x - sqrt_d);
       
    // compute max cubic root from three real roots using complex number formula eq(0.14)  
    // revert transformation eq(0.2) and eq(0.16)
    float y3_max = cos(atan(sqrt_d, -r.x) / 3.0);
    y3_max *= sqrt(max(-r.y, 0.0)); 

    // choose cubic roots based on discriminant sign 
    float y_max = (d < 0.0) ? y3_max : y1;

    // return root
    return y_max;
}

// Solve subsidiary quadratics 
// g^2 + p1g + q1 = 0
// h^2 + p2h + q2 = 0
vec4 subsidiary_quadratics_roots(vec4 p1_p2_q1_q2)
{
    // Solve in parallel the factored quadratics from the quartic 
    p1_p2_q1_q2.xy /= -2.0;
    
    // compute the fused quadratic discriminants
    vec2 d = p1_p2_q1_q2.xy * p1_p2_q1_q2.xy - p1_p2_q1_q2.zw;
    vec2 sqrt_d = sqrt(max(d, 0.0));

    // compute first roots
    vec2 g1_h1 = vec2(
        p1_p2_q1_q2.x + sqrt_d.x * ssign(p1_p2_q1_q2.x), 
        p1_p2_q1_q2.y + sqrt_d.y * ssign(p1_p2_q1_q2.y)
    );

    // compute rest of the roots via stable formula
    vec2 g2_h2 = p1_p2_q1_q2.zw / g1_h1;

    // combine roots
    return vec4(g1_h1.x, g2_h2.x, g1_h1.y, g2_h2.y);
}

// Solve the pair of factored quadratics in parallel
// x^2 + g1x + h1 = 0
// x^2 + g2x + h2 = 0
vec4 factored_quadratics_roots(vec4 g1_g2_h1_h2)
{
    // Solve in parallel the factored quadratics from the quartic 
    g1_g2_h1_h2.xy /= -2.0;
    
    // compute the fused quadratic discriminants
    vec2 d = g1_g2_h1_h2.xy * g1_g2_h1_h2.xy - g1_g2_h1_h2.zw;
    vec2 sqrt_d = sqrt(max(d, 0.0));
    
    // since we know the signs of s, u
    vec2 x1_x2 = vec2(
        g1_g2_h1_h2.x + sqrt_d.x * ssign(g1_g2_h1_h2.x), 
        g1_g2_h1_h2.y + sqrt_d.y * ssign(g1_g2_h1_h2.y)
    );

    // compute rest of the roots via stable formula
    vec2 x3_x4 = g1_g2_h1_h2.zw / x1_x2;

    // compute rest of the roots via stable formula
    return vec4(x1_x2, x3_x4);
}

// Solve quartic equation c0 + c1x^1 + c2x^2 + c3x^3 + c4x^4 = 0 
// using Ferrari-Descartes method assuming quartic coefficient is nonzero

vec4 quartic_roots_new(in float c[5]) 
{
    // To simplify depressed quartic computations
    // d + cx^1 + 3bx^2 + ax^3 + x^4
    vec4 n = vec4(c[0], c[1], c[2], c[3]) / c[4];
    n.z /= 3.0;

    // Compute resolvent depressed cubic coefficients 
    // rc + rb * ys + ys^3
    float bb = n.z * n.z;
    float rb, rc;

    // rb = (ac - 4d) - 3d^2
    rb = dot(vec2(n.w, -4.0), n.yx);
    rb -= bb * 3.0;

    // rc = 12bd - a^2d - c^2 + (ac - 4d)b - 2b^3
    rc = dot(vec2(12.0, -n.w), n.zw);
    rc = dot(vec2(rc,   -n.y), n.xy);
    rc += (rb - bb) * n.z;

    // Solve resolvent depressed cubic for max root and shift solution
    // rc + rb * ys + ys^3 = 0 and y = ys + b
    float y_max = resolvent_depressed_cubic_max_root(rc, rb);
    y_max += n.z;

    // Compute subsidiary quadratics coefficients
    vec4 p1_p2_q1_q2 = vec4(-n.w,-y_max, n.z * 3.0 - y_max, n.x);

    // Solve subsidiary quadratics resulting from cubic solution
    // g^2 + p1g + q1 = 0 and h^2 + p2h + q2 = 0
    vec4 g1_g2_h1_h2 = subsidiary_quadratics_roots(p1_p2_q1_q2);

    // Solve the pair of factored quadratics in parallel
    // x^2 + g1x + h1 = 0 and x^2 + g2x + h2 = 0
    vec4 x1_x2_x3_x4 = factored_quadratics_roots(g1_g2_h1_h2);

    // Return solutions
    return x1_x2_x3_x4;
}



#endif