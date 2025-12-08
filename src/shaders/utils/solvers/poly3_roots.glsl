
/* Soures
Finding Real Polynomial Roots on GPUs (https://momentsingraphics.de/GPUPolynomialRoots.html),
Shadertoy Spherical harmonics glyphs (https://www.shadertoy.com/view/dlGSDV),
Ray Tracing Spherical Harmonics Glyphs (https://momentsingraphics.de/VMV2023.html),
High-Performance Polynomial Solver Cem Yuksel (https://www.cemyuksel.com/research/polynomials/),
cyPolynomial.h class (https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h),
*/

#ifndef POLY3_ROOTS
#define POLY3_ROOTS

// When there are fewer intersections/roots than theoretically possible, some
// array entries are set to this value
#define POLY3_NO_INTERSECTION 3.4e38

// Searches a single root of a polynomial within a given interval.
// \param out_root The location of the found root.
// \param out_end_value The value of the given polynomial at end.
// \param poly Coefficients of the polynomial for which a root should be found.
//        Coefficient poly[i] is multiplied by x^i.
// \param begin The beginning of an interval where the polynomial is monotonic.
// \param end The end of said interval.
// \param begin_value The value of the given polynomial at begin.
// \param error_tolerance The error tolerance for the returned root location.
//        Typically the error will be much lower but in theory it can be
//        bigger.
// \return true if a root was found, false if no root exists.
bool poly3_roots_newton_bisection
(
    out float out_root, 
    out float out_end_value,
    float poly3[4], 
    float begin, 
    float end,
    float begin_value, 
    float error_tolerance
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

    // Otherwise, we find the root iteratively using Newton bisection (with
    // bounded iteration count)
    float current = 0.5 * (begin + end);

    #pragma no_unroll
    for (int i = 0; i != 10; ++i) 
    {
        // Evaluate the polynomial and its derivative
        float derivative = poly3[3];
        float value = poly3[3] * current + poly3[2];
        #pragma unroll
        for (int j = 1; j != -1; --j) 
        {
            derivative = derivative * current + value;
            value = value * current + poly3[j];
        }
    
        // Shorten the interval
        bool right = begin_value * value > 0.0;
        begin = right ? current : begin;
        end = right ? end : current;

        // Apply Newton's method
        float guess = current - value / derivative;

        // Pick a guess
        float middle = 0.5 * (begin + end);
        float next = (guess >= begin && guess <= end) ? guess : middle;

        // Move along or terminate
        bool done = abs(next - current) < error_tolerance;
        current = next;
        if (done) break;
    }

    out_root = current;
    return true;
}

// Finds all roots of the given polynomial in the interval [begin, end] and
// writes them to out_roots. Some entries will be POLY3_NO_INTERSECTION but other 
// than that the array is sorted. The last entry is always POLY3_NO_INTERSECTION.
void poly3_roots
(
    out vec4 out_roots, 
    vec4 poly3, 
    float begin, 
    float end
){
    float tolerance = (end - begin) * 1.0e-9;

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
        out_roots[1] = min(root_0, root_1);
        out_roots[2] = max(root_0, root_1);
    }
    else 
    {
        // Indicate that the cubic derivative has a single root
        out_roots[1] = begin;
        out_roots[2] = begin;
    }

    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched roots are set to begin
    out_roots[0] = begin;
    out_roots[3] = end;

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
    #pragma unroll
    for (int i = 0; i != 3; ++i) 
    {
        float current_begin = out_roots[i];
        float current_end = out_roots[i + 1];

        // Try to find a root
        float root;
        if (poly3_roots_newton_bisection(root, begin_value, derivative, current_begin, current_end, begin_value, tolerance))
        {
            out_roots[i] = root;
        }
        else
        {
            out_roots[i] = POLY3_NO_INTERSECTION;
        }
    }
 
    // We no longer need this array entry
    out_roots[3] = POLY3_NO_INTERSECTION;
}

#endif