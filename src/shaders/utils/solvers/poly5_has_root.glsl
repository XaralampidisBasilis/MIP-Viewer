
/* Soures
Finding Real Polynomial Roots on GPUs (https://momentsingraphics.de/GPUPolynomialRoots.html),
Shadertoy Spherical harmonics glyphs (https://www.shadertoy.com/view/dlGSDV),
Ray Tracing Spherical Harmonics Glyphs (https://momentsingraphics.de/VMV2023.html),
High-Performance Polynomial Solver Cem Yuksel (https://www.cemyuksel.com/research/polynomials/),
cyPolynomial.h class (https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h),
*/

#ifndef POLY5_HAS_ROOT
#define POLY5_HAS_ROOT

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

bool poly5_has_root_newton_bisection
(
    out float out_root, 
    out float out_end_value,
    float poly[6], 
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
    out_end_value = poly[5];
    out_end_value = out_end_value * end + poly[4];
    out_end_value = out_end_value * end + poly[3];
    out_end_value = out_end_value * end + poly[2];
    out_end_value = out_end_value * end + poly[1];
    out_end_value = out_end_value * end + poly[0];

    // If the values at both ends have the same non-zero sign, there is no root
    if ((begin_value > 0.0) == (out_end_value > 0.0)) return false;

    // Otherwise, we find the root iteratively using Newton bisection (with
    // bounded iteration count)
    float current = 0.5 * (begin + end);

    #pragma no_unroll
    for (int i = 0; i != 50; ++i) 
    {
        // Evaluate the polynomial and its derivative
        float derivative = poly[5];
        float value = derivative * current + poly[4];
        derivative = derivative * current + value;
        value = value * current + poly[3];
        derivative = derivative * current + value;
        value = value * current + poly[2];
        derivative = derivative * current + value;
        value = value * current + poly[1];
        derivative = derivative * current + value;
        value = value * current + poly[0];

        // Shorten the interval
        bool right = (begin_value > 0.0) == (value > 0.0);
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

// Searches a single root of a polynomial within a given interval.
// \param out_end_value The value of the given polynomial at end.
// \param poly Coefficients of the polynomial for which a root should be found.
//        Coefficient poly[i] is multiplied by x^i.
// \param begin The beginning of an interval where the polynomial is monotonic.
// \param end The end of said interval.
// \param begin_value The value of the given polynomial at begin.
// \return true if a root was found, false if no root exists.
bool poly5_has_root_sign_change
(
    out float out_end_value,
    in float poly[6], 
    in float begin, 
    in float end,
    in float begin_value
){
    if (begin == end) 
    {
        out_end_value = begin_value;
        return false;
    }

    // Evaluate the polynomial at the end of the interval
    out_end_value = poly[5];
    out_end_value = out_end_value * end + poly[4];
    out_end_value = out_end_value * end + poly[3];
    out_end_value = out_end_value * end + poly[2];
    out_end_value = out_end_value * end + poly[1];
    out_end_value = out_end_value * end + poly[0];

    // If the values at both ends have the same non-zero sign, there is no root
    if ((begin_value > 0.0) == (out_end_value > 0.0)) return false;

    return true;
}

// Finds if the given polynomial has root in the interval [begin, end]
bool poly5_has_root
(
    float poly[6], 
    float begin, 
    float end
){
    float tolerance = (end - begin) * 1.0e-6;

    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched critical roots are set to begin
    float critical_roots[6];
    critical_roots[0] = begin;
    critical_roots[1] = begin;
    critical_roots[2] = begin;
    critical_roots[5] = end;

    // Construct the quadratic derivative of the polynomial. We divide each
    // derivative by the factorial of its order = 3, such that the constant
    // coefficient can be copied directly from poly. That is a safeguard
    // against overflow and makes it easier to avoid spilling below. The
    // factors happen to be binomial coefficients then.
    float deriv_poly[6];
    deriv_poly[5] = 0.0;
    deriv_poly[4] = 0.0;
    deriv_poly[3] = 0.0;
    deriv_poly[2] = poly[5] * 10.0; // 5*4*3 / 3!
    deriv_poly[1] = poly[4] * 4.0;  // 4*3*2 / 3!
    deriv_poly[0] = poly[3];        //   3*2 / 3!
    
    // Compute its two roots using the quadratic formula
    float discriminant = deriv_poly[1] * deriv_poly[1] - 4.0 * deriv_poly[0] * deriv_poly[2];
    if (discriminant >= 0.0) 
    {
        float sqrt_discriminant = sqrt(discriminant);
        float scaled_root = deriv_poly[1] + (deriv_poly[1] > 0.0 ? sqrt_discriminant : -sqrt_discriminant);
        float root0 = -2.0 * deriv_poly[0] / scaled_root;
        float root1 = -0.5 * scaled_root / deriv_poly[2];
        root0 = clamp(root0, begin, end);
        root1 = clamp(root1, begin, end);

        critical_roots[3] = min(root0, root1);
        critical_roots[4] = max(root0, root1);
    }
    else 
    {
        // Indicate that the cubic derivative has a single root
        critical_roots[3] = begin;
        critical_roots[4] = begin;
    }

    // Work your way up to derivatives of higher degree until you reach the
    // polynomial itself. This implementation may seem peculiar: It always
    // treats the derivative as though it had degree 5 and it
    // constructs the derivatives in a contrived way. Changing that would
    // reduce the number of arithmetic instructions roughly by a factor of two.
    // However, it would also cause register spilling, which has a far more
    // negative impact on the overall run time. Profiling indicates that the
    // current implementation has no spilling whatsoever.    
  
    // degree = 3
    // Take the integral of the previous derivative (scaled such that the
    // constant coefficient can still be copied directly from poly)
    deriv_poly[5] = 0.0;
    deriv_poly[4] = 0.0;
    deriv_poly[3] = deriv_poly[2] * (3.0 / 3.0);
    deriv_poly[2] = deriv_poly[1] * (3.0 / 2.0);
    deriv_poly[1] = deriv_poly[0] * (3.0 / 1.0);
    deriv_poly[0] = poly[2];

    // Determine the value of this derivative at begin
    float begin_value = deriv_poly[5];
    begin_value = begin_value * begin + deriv_poly[4];
    begin_value = begin_value * begin + deriv_poly[3];
    begin_value = begin_value * begin + deriv_poly[2];
    begin_value = begin_value * begin + deriv_poly[1];
    begin_value = begin_value * begin + deriv_poly[0];

    // Iterate over the intervals where roots may be found
    #pragma unroll
    for (int i = 2; i != 5; ++i) 
    {
        float root;
        if (poly5_has_root_newton_bisection(root, begin_value, deriv_poly, critical_roots[i], critical_roots[i + 1], begin_value, tolerance))
        {
            critical_roots[i] = root;
        }
        else
        {
            // Create an empty interval for the next iteration
            critical_roots[i] = critical_roots[i - 1];
        }
    }

    // degree = 4
    // Take the integral of the previous derivative (scaled such that the
    // constant coefficient can still be copied directly from poly)
    deriv_poly[5] = 0.0;
    deriv_poly[4] = deriv_poly[3] * (2.0 / 4.0);
    deriv_poly[3] = deriv_poly[2] * (2.0 / 3.0);
    deriv_poly[2] = deriv_poly[1] * (2.0 / 2.0);
    deriv_poly[1] = deriv_poly[0] * (2.0 / 1.0);
    deriv_poly[0] = poly[1];

    // Determine the value of this derivative at begin
    begin_value = deriv_poly[5];
    begin_value = begin_value * begin + deriv_poly[4];
    begin_value = begin_value * begin + deriv_poly[3];
    begin_value = begin_value * begin + deriv_poly[2];
    begin_value = begin_value * begin + deriv_poly[1];
    begin_value = begin_value * begin + deriv_poly[0];

    // Iterate over the intervals where roots may be found
    #pragma unroll
    for (int i = 1; i != 5; ++i) 
    {
        // Try to find a root
        float root;
        if (poly5_has_root_newton_bisection(root, begin_value, deriv_poly, critical_roots[i], critical_roots[i + 1], begin_value, tolerance))
        {
            critical_roots[i] = root;
        }
        else
        {
            // Create an empty interval for the next iteration
            critical_roots[i] = critical_roots[i - 1];
        }
    }
    
    // degree = 5
    // Determine the value of this derivative at begin
    begin_value = poly[5];
    begin_value = begin_value * begin + poly[4];
    begin_value = begin_value * begin + poly[3];
    begin_value = begin_value * begin + poly[2];
    begin_value = begin_value * begin + poly[1];
    begin_value = begin_value * begin + poly[0];

    // Iterate over the intervals where sign change may be found
    #pragma unroll
    for (int i = 0; i != 5; ++i) 
    {
        // Try to find sign change
        if (poly5_has_root_sign_change(begin_value, poly, critical_roots[i], critical_roots[i + 1], begin_value))
        {
            return true;
        }
    }
 
    return false;
}

// Finds if the given polynomial has root in the interval [begin, end]
// using cubic reduction to quadratic
bool poly5_has_root_v2
(
    float poly[6], 
    float begin, 
    float end
){
    float tolerance = (end - begin) * 1.0e-6;

    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched critical roots are set to begin
    float critical_roots[6];
    critical_roots[0] = begin;
    critical_roots[1] = begin;
    critical_roots[2] = begin;
    critical_roots[5] = end;

    // Construct the quadratic derivative of the polynomial. We divide each
    // derivative by the factorial of its order = 3, such that the constant
    // coefficient can be copied directly from poly. That is a safeguard
    // against overflow and makes it easier to avoid spilling below. The
    // factors happen to be binomial coefficients then.
    float deriv_poly[6];
    deriv_poly[5] = 0.0;
    deriv_poly[4] = 0.0;
    deriv_poly[3] = 0.0;
    deriv_poly[2] = poly[5] * 10.0; // 5*4*3 / 3!
    deriv_poly[1] = poly[4] * 4.0;  // 4*3*2 / 3!
    deriv_poly[0] = poly[3];        //   3*2 / 3!

    // Compute its two roots using the quadratic formula
    float discriminant = deriv_poly[1] * deriv_poly[1] - 4.0 * deriv_poly[0] * deriv_poly[2];
    if (discriminant >= 0.0) 
    {
        float sqrt_discriminant = sqrt(discriminant);
        float scaled_root = deriv_poly[1] + (deriv_poly[1] > 0.0 ? sqrt_discriminant : -sqrt_discriminant);
        float root0 = -2.0 * deriv_poly[0] / scaled_root;
        float root1 = -0.5 * scaled_root / deriv_poly[2];
        root0 = clamp(root0, begin, end);
        root1 = clamp(root1, begin, end);

        critical_roots[3] = min(root0, root1);
        critical_roots[4] = max(root0, root1);
    }
    else 
    {
        // Indicate that the cubic derivative has a single root
        critical_roots[3] = begin;
        critical_roots[4] = end;
    }

    // Work your way up to derivatives of higher degree until you reach the
    // polynomial itself. This implementation may seem peculiar: It always
    // treats the derivative as though it had degree 5 and it
    // constructs the derivatives in a contrived way. Changing that would
    // reduce the number of arithmetic instructions roughly by a factor of two.
    // However, it would also cause register spilling, which has a far more
    // negative impact on the overall run time. Profiling indicates that the
    // current implementation has no spilling whatsoever.    

    // degree = 3
    // Take the integral of the previous derivative (scaled such that the
    // constant coefficient can still be copied directly from poly)
    deriv_poly[5] = 0.0;
    deriv_poly[4] = 0.0;
    deriv_poly[3] = deriv_poly[2] * (3.0 / 3.0);
    deriv_poly[2] = deriv_poly[1] * (3.0 / 2.0);
    deriv_poly[1] = deriv_poly[0] * (3.0 / 1.0);
    deriv_poly[0] = poly[2];

    float begin_value = deriv_poly[5];
    begin_value = begin_value * begin + deriv_poly[4];
    begin_value = begin_value * begin + deriv_poly[3];
    begin_value = begin_value * begin + deriv_poly[2];
    begin_value = begin_value * begin + deriv_poly[1];
    begin_value = begin_value * begin + deriv_poly[0];

    float root0;
    if (poly5_has_root_newton_bisection(root0, begin_value, deriv_poly, critical_roots[3], critical_roots[4], begin_value, tolerance))
    {
        // If you find a root deflate the cubic to quadratic and solve analytically
        deriv_poly[4] = deriv_poly[2] + root0 * deriv_poly[3];
        deriv_poly[5] = deriv_poly[1] + root0 * deriv_poly[4];

        // Compute its two roots using the quadratic formula
        float discriminant = deriv_poly[4] * deriv_poly[4] - 4.0 * deriv_poly[5] * deriv_poly[3];
        if (discriminant >= 0.0) 
        {
            float sqrt_discriminant = sqrt(discriminant);
            float scaled_root = deriv_poly[4] + (deriv_poly[4] > 0.0 ? sqrt_discriminant : -sqrt_discriminant);
            float root1 = -2.0 * deriv_poly[5] / scaled_root;
            float root2 = -0.5 * scaled_root / deriv_poly[3];
            root1 = clamp(root1, begin, end);
            root2 = clamp(root2, begin, end);
            
            // 3-element sorting network
            if (root0 > root1) { float t = root0; root0 = root1; root1 = t; }
            if (root0 > root2) { float t = root0; root0 = root2; root2 = t; }
            if (root1 > root2) { float t = root1; root1 = root2; root2 = t; }

            // Indicate that the quartic derivative has three roots
            critical_roots[2] = root0;
            critical_roots[3] = root1;
            critical_roots[4] = root2;
        }
        else
        {
            // Indicate that the quartic derivative has two roots
            critical_roots[2] = begin;
            critical_roots[3] = begin;
            critical_roots[4] = root0;
        }
    }
    else
    {
        // Indicate that the quartic derivative has one  root
        critical_roots[2] = begin;
        critical_roots[3] = begin;
        critical_roots[4] = begin;
    }

    // degree = 4
    // Take the integral of the previous derivative (scaled such that the
    // constant coefficient can still be copied directly from poly)
    deriv_poly[5] = 0.0;
    deriv_poly[4] = deriv_poly[3] * (2.0 / 4.0);
    deriv_poly[3] = deriv_poly[2] * (2.0 / 3.0);
    deriv_poly[2] = deriv_poly[1] * (2.0 / 2.0);
    deriv_poly[1] = deriv_poly[0] * (2.0 / 1.0);
    deriv_poly[0] = poly[1];

    // Determine the value of this derivative at begin
    begin_value = deriv_poly[5];
    begin_value = begin_value * begin + deriv_poly[4];
    begin_value = begin_value * begin + deriv_poly[3];
    begin_value = begin_value * begin + deriv_poly[2];
    begin_value = begin_value * begin + deriv_poly[1];
    begin_value = begin_value * begin + deriv_poly[0];

    // Iterate over the intervals where roots may be found
    #pragma unroll
    for (int i = 1; i != 5; ++i) 
    {
        // Try to find a root
        float root;
        if (poly5_has_root_newton_bisection(root, begin_value, deriv_poly, critical_roots[i], critical_roots[i + 1], begin_value, tolerance))
        {
            critical_roots[i] = root;
        }
        else
        {
            // Create an empty interval for the next iteration
            critical_roots[i] = critical_roots[i - 1];
        }
    }
    
    // degree = 5
    // Determine the value of this derivative at begin
    begin_value = poly[5];
    begin_value = begin_value * begin + poly[4];
    begin_value = begin_value * begin + poly[3];
    begin_value = begin_value * begin + poly[2];
    begin_value = begin_value * begin + poly[1];
    begin_value = begin_value * begin + poly[0];

    // Iterate over the intervals where sign change may be found
    #pragma unroll
    for (int i = 0; i != 5; ++i) 
    {
        // Try to find sign change
        if (poly5_has_root_sign_change(begin_value, poly, critical_roots[i], critical_roots[i + 1], begin_value))
        {
            return true;
        }
    }
 
    return false;
}

#endif