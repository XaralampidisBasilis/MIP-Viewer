
/* Soures
Finding Real Polynomial Roots on GPUs (https://momentsingraphics.de/GPUPolynomialRoots.html),
Shadertoy Spherical harmonics glyphs (https://www.shadertoy.com/view/dlGSDV),
Ray Tracing Spherical Harmonics Glyphs (https://momentsingraphics.de/VMV2023.html),
High-Performance Polynomial Solver Cem Yuksel (https://www.cemyuksel.com/research/polynomials/),
cyPolynomial.h class (https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h),
*/

#ifndef CUBIC_HAS_ROOT
#define CUBIC_HAS_ROOT

#ifndef CUBIC_SAMPLES_SUBDIVS
#define CUBIC_SAMPLES_SUBDIVS 2
#endif

// Searches a single root of a polynomial within a given interval.
// \param out_end_value The value of the given polynomial at end.
// \param poly Coefficients of the polynomial for which a root should be found.
//        Coefficient poly[i] is multiplied by x^i.
// \param begin The beginning of an interval where the polynomial is monotonic.
// \param end The end of said interval.
// \param begin_value The value of the given polynomial at begin.
// \return true if a root was found, false if no root exists.
bool cubic_sign_change
(
    out float out_end_value,
    vec4 poly, 
    float begin, 
    float end,
    float begin_value
){
    if (begin == end) 
    {
        out_end_value = begin_value;
        return false;
    }

    // Evaluate the polynomial at the end of the interval
    out_end_value = poly[3];
    out_end_value = out_end_value * end + poly[2];
    out_end_value = out_end_value * end + poly[1];
    out_end_value = out_end_value * end + poly[0];

    return (begin_value * out_end_value <= 0.0);
}

// Finds if the given polynomial has root in the interval [begin, end]
bool cubic_has_root(
    vec4 poly, 
    float begin, 
    float end
){
    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched roots are set to begin
    vec4 crit_roots;
    crit_roots[0] = begin;
    crit_roots[3] = end;

    // Construct the quadratic derivative of the polynomial. We divide each
    // derivative by the factorial of its order, such that the constant
    // coefficient can be copied directly from poly. That is a safeguard
    // against overflow and makes it easier to avoid spilling below. The
    // factors happen to be binomial coefficients then.
    vec4 deriv_poly;
    deriv_poly[0] = poly[1];
    deriv_poly[1] = poly[2] * 2.0;
    deriv_poly[2] = poly[3] * 3.0;
    deriv_poly[3] = 0.0;

    // Compute its two roots using the quadratic formula
    float discriminant = deriv_poly[1] * deriv_poly[1] - 4.0 * deriv_poly[0] * deriv_poly[2];
    if (discriminant >= 0.0) 
    {
        // Compute the quadratic roots using numerically stable solutions
        float sqrt_disc = sqrt(discriminant);
        float scaled_root = -0.5 * (deriv_poly[1] + sqrt_disc * sign(deriv_poly[1]));
        float root_0 = clamp(deriv_poly[0] / scaled_root, begin, end);
        float root_1 = clamp(scaled_root / deriv_poly[2], begin, end); 

        crit_roots[1] = min(root_0, root_1);
        crit_roots[2] = max(root_0, root_1);
    }
    else
    {
        crit_roots[1] = begin;
        crit_roots[2] = begin;
    }

    // Determine the value of this deriv_poly at begin
    float begin_value = poly[3];
    begin_value = begin_value * begin + poly[2];
    begin_value = begin_value * begin + poly[1];
    begin_value = begin_value * begin + poly[0];

    // Iterate over the intervals where sign change may be found
    #pragma unroll
    for (int i = 0; i <= 2; ++i) 
    {
        float current_begin = crit_roots[i];
        float current_end = crit_roots[i + 1];

        // Try to find sign change
        if (cubic_sign_change(begin_value, poly, current_begin, current_end, begin_value))
        {
            return true;
        }
    };

    return false;
}

// Finds if the given cubic polynomial has root in the interval [begin, end]
// via uniform sampling in 4-wide tiles (SIMD). Checks sign changes.
bool cubic_has_root_sample(
    vec4 poly, 
    const float begin, 
    const float end
){
    float delta = (end - begin) / float(CUBIC_SAMPLES_SUBDIVS * 4);
    float step = delta * 4.0;

    // Start previous value at begin
    float prev = ((poly[3]*begin + poly[2])*begin + poly[1])*begin + poly[0];

    // First tile positions at begin
    vec4 pos = begin + delta * vec4(1.0, 2.0, 3.0, 4.0);

    #pragma unroll
    for (int i = 0; i < CUBIC_SAMPLES_SUBDIVS; ++i) 
    {
        // Horner on 4 positions at once
        vec4 v = vec4(poly[3]);
        v = v * pos + poly[2];
        v = v * pos + poly[1];
        v = v * pos + poly[0];

        // sign-change test across the 4 consecutive edges:
        vec4 a = vec4(prev, v.xyz);
        vec4 b = v;

        // opposite signs or exactly zero
        bool has_root = any(lessThanEqual(a * b, vec4(0.0)));
        if (has_root) return true;

        prev = v.w;  // carry last sample into next edge
        pos += step; // next tile (advance by 4 segments)
    }

    return false;
}

#endif