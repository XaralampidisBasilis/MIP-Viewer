
/* Soures
Finding Real Polynomial Roots on GPUs (https://momentsingraphics.de/GPUPolynomialRoots.html),
Shadertoy Spherical harmonics glyphs (https://www.shadertoy.com/view/dlGSDV),
Ray Tracing Spherical Harmonics Glyphs (https://momentsingraphics.de/VMV2023.html),
High-Performance Polynomial Solver Cem Yuksel (https://www.cemyuksel.com/research/polynomials/),
cyPolynomial.h class (https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h),
*/

#ifndef POLY3_HAS_ROOT
#define POLY3_HAS_ROOT

// Searches a single root of a polynomial within a given interval.
// \param out_end_value The value of the given polynomial at end.
// \param poly Coefficients of the polynomial for which a root should be found.
//        Coefficient poly[i] is multiplied by x^i.
// \param begin The beginning of an interval where the polynomial is monotonic.
// \param end The end of said interval.
// \param begin_value The value of the given polynomial at begin.
// \return true if a root was found, false if no root exists.
bool poly3_has_root_sign_change
(
    out float out_end_value,
    float poly3[4], 
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
    out_end_value = poly3[3];
    out_end_value = out_end_value * end + poly3[2];
    out_end_value = out_end_value * end + poly3[1];
    out_end_value = out_end_value * end + poly3[0];

    // If the values at both ends have the same non-zero sign, there is no root
    if (begin_value * out_end_value > 0.0) return false;

    return true;
}


// Finds if the given polynomial has root in the interval [begin, end]
bool poly3_has_root
(
    vec4 poly3, 
    float begin, 
    float end
){

    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched roots are set to begin
    vec4 critical_roots = vec4(
        begin,
        begin, // Indicate that the cubic derivative has a single root
        begin, // Indicate that the cubic derivative has a single root
        end
    );

    // Construct the quadratic derivative of the polynomial. We divide each
    // derivative by the factorial of its order, such that the constant
    // coefficient can be copied directly from poly. That is a safeguard
    // against overflow and makes it easier to avoid spilling below. The
    // factors happen to be binomial coefficients then.
    float derivative[4];
    derivative[0] = poly3[1];
    derivative[1] = poly3[2] * 2.0;
    derivative[2] = poly3[3] * 3.0;
    derivative[3] = 0.0;

    // Compute its two roots using the quadratic formula
    float discriminant = derivative[1] * derivative[1] - 4.0 * derivative[0] * derivative[2];

    if (discriminant >= 0.0) 
    {
        float sqrt_discriminant = sqrt(discriminant);
        float scaled_root = derivative[1] + ((derivative[1] > 0.0) ? sqrt_discriminant : (-sqrt_discriminant));
        float root_0 = clamp(-2.0 * derivative[0] / scaled_root, begin, end);
        float root_1 = clamp(-0.5 * scaled_root / derivative[2], begin, end);
        critical_roots[1] = min(root_0, root_1);
        critical_roots[2] = max(root_0, root_1);
    }

    // Work your way up to derivatives of higher degree until you reach the
    // polynomial itself. This implementation may seem peculiar: It always
    // treats the derivative as though it had degree 3 and it
    // constructs the derivatives in a contrived way. Changing that would
    // reduce the number of arithmetic instructions roughly by a factor of two.
    // However, it would also cause register spilling, which has a far more
    // negative impact on the overall run time. Profiling indicates that the
    // current implementation has no spilling whatsoever.

    // Take the integral of the previous derivative (scaled such that the
    // constant coefficient can still be copied directly from poly)
    // Copy the constant coefficient without causing spilling. This part
    // would be harder if the derivative were not scaled the way it is.
    derivative[3] = derivative[2] / 3.0;
    derivative[2] = derivative[1] / 2.0;
    derivative[1] = derivative[0];
    derivative[0] = poly3[0];

    // Determine the value of this derivative at begin
    float begin_value = derivative[3];
    begin_value = begin_value * begin + derivative[2];
    begin_value = begin_value * begin + derivative[1];
    begin_value = begin_value * begin + derivative[0];

    // Iterate over the intervals where roots may be found
    if (poly3_has_root_sign_change(begin_value, derivative, critical_roots[0], critical_roots[1], begin_value)) return true;
    if (poly3_has_root_sign_change(begin_value, derivative, critical_roots[1], critical_roots[2], begin_value)) return true;
    if (poly3_has_root_sign_change(begin_value, derivative, critical_roots[2], critical_roots[3], begin_value)) return true;
        

    return false;
}

#endif