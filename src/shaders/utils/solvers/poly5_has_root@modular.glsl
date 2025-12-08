
/* Soures
Finding Real Polynomial Roots on GPUs (https://momentsingraphics.de/GPUPolynomialRoots.html),
Shadertoy Spherical harmonics glyphs (https://www.shadertoy.com/view/dlGSDV),
Ray Tracing Spherical Harmonics Glyphs (https://momentsingraphics.de/VMV2023.html),
High-Performance Polynomial Solver Cem Yuksel (https://www.cemyuksel.com/research/polynomials/),
cyPolynomial.h class (https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h),
*/

#ifndef POLY5_HAS_ROOT
#define POLY5_HAS_ROOT

#ifndef MICRO_TOLERANCE
#define MICRO_TOLERANCE 1.0e-6
#endif
#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif
#ifndef SIGN_CHANGE
#include "../math/sign_change"
#endif
#ifndef SWAP
#include "../math/swap"
#endif


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

bool poly5_has_root_newton_bisection(
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
    out_end_value = eval_poly(poly, end);

    // If the values at both ends have the same non-zero sign, there is no root
    if (!sign_change(begin_value, out_end_value)) return false;

    // Otherwise, we find the root iteratively using Newton bisection (with
    // bounded iteration count)
    float current = 0.5 * (begin + end);

    #pragma no_unroll
    for (int i = 0; i != 50; ++i) 
    {
        // Evaluate the polynomial and its derivative
        float derivative, value = eval_poly(poly, current, derivative);

        // Shorten the interval
        bool left = sign_change(begin_value, value);
        begin = left ? begin : current;
        end = left ? current : end;

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
bool poly5_has_root_sign_change(
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
    out_end_value = eval_poly(poly, end);

    // If the values at both ends have the same non-zero sign, there is no root
    return sign_change(begin_value, out_end_value);
}

// Computes the roots of the quadratic derivative of the polynomial.
// \param poly Coefficients of the polynomial for which a root should be found.
//        Coefficient poly[i] is multiplied by x^i.
// \param begin The beginning of search interval.
// \param end The end of search interval.
// \return vec2 the quadratic roots found in [begin, end].
vec2 poly5_has_root_quadratic_roots(
    in float poly[6], 
    in float begin, 
    in float end,
    in float tolerance
){
    // Indicate that the cubic derivative has a single root
    vec2 out_roots = vec2(begin);
    float discriminant = poly[1] * poly[1] - 4.0 * poly[0] * poly[2];

    if (discriminant >= 0.0) 
    {
        float sqrt_discriminant = sqrt(max(discriminant, 0.0));
        float scaled_root = poly[1] + (poly[1] > 0.0 ? sqrt_discriminant : -sqrt_discriminant);
        float root0 = -2.0 * poly[0] / scaled_root;
        float root1 = -0.5 * scaled_root / poly[2];
        // root1 = (abs(poly[2]) < tolerance) ?  root0 : root1;
        
        // Indicate that the cubic derivative has two roots
        out_roots[0] = min(root0, root1);
        out_roots[1] = max(root0, root1);

        out_roots = clamp(out_roots, begin, end);
    }

    return out_roots;
}

// Computes the roots of the cubic derivative of the polynomial using deflation.
// \param poly Coefficients of the polynomial for which a root should be found.
//        Coefficient poly[i] is multiplied by x^i.
// \param begin The beginning of search interval.
// \param end The end of search interval.
// \param begin_root The beginning of an interval where the polynomial is monotonic.
// \param end_root The end of said interval.    
// \return vec3 the cubic roots found in [begin, end].
vec3 poly5_has_root_cubic_roots(
    in float poly[6], 
    in float begin, 
    in float end,
    in float begin_root,
    in float end_root,
    in float tolerance
){
    vec3 out_roots = vec3(begin);
    float begin_value = eval_poly(poly, begin);

    float root0;
    if (poly5_has_root_newton_bisection(root0, begin_value, poly, begin_root, end_root, begin_value, tolerance))
    {
        // If you find a root deflate the cubic to quadratic and solve analytically
        poly[4] = poly[2] + root0 * poly[3];
        poly[5] = poly[1] + root0 * poly[4];

        // Compute its two roots using the quadratic formula
        float discriminant = poly[4] * poly[4] - 4.0 * poly[5] * poly[3];

        if (discriminant >= 0.0) 
        {
            float sqrt_discriminant = sqrt(max(discriminant, 0.0));
            float scaled_root = poly[4] + (poly[4] > 0.0 ? sqrt_discriminant : -sqrt_discriminant);
            float root1 = -2.0 * poly[5] / scaled_root;
            float root2 = -0.5 * scaled_root / poly[3];
            // root2 = (abs(poly[3]) < tolerance) ?  root1 : root2;

            // 3-element sorting network
            if (root0 > root1) { swap(root0, root1); }
            if (root0 > root2) { swap(root0, root2); }
            if (root1 > root2) { swap(root1, root2); }

            // Indicate that the quartic derivative has three roots
            out_roots[0] = root0;
            out_roots[1] = root1;
            out_roots[2] = root2;

            // Clamp roots to interval
            out_roots = clamp(out_roots, begin, end);
        }
        else
        {
            // Indicate that the quartic derivative has two roots
            out_roots[2] = root0;
        }
    }
    
    return out_roots;
}

// Finds if the given polynomial has root in the interval [begin, end]
bool poly5_has_root(
    float poly[6], 
    float begin, 
    float end
){
    float tolerance = (end - begin) * MICRO_TOLERANCE;

    // Construct the quadratic derivative of the polynomial. We divide each
    // derivative by the factorial of its order = 3, such that the constant
    // coefficient can be copied directly from poly. That is a safeguard
    // against overflow and makes it easier to avoid spilling below. The
    // factors happen to be binomial coefficients then.
    float deriv_poly[6];
    deriv_poly[5] = 0.0;
    deriv_poly[4] = 0.0;
    deriv_poly[3] = 0.0;
    deriv_poly[2] = poly[5] * 10.0; 
    deriv_poly[1] = poly[4] * 4.0;  
    deriv_poly[0] = poly[3];        
    
    // Compute its two roots using the quadratic formula
    vec2 quad_roots = poly5_has_root_quadratic_roots(deriv_poly, begin, end, tolerance);

    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched critical roots are set to begin
    float critical_roots[6];
    critical_roots[0] = begin;
    critical_roots[1] = begin;
    critical_roots[2] = begin;
    critical_roots[3] = quad_roots[0];
    critical_roots[4] = quad_roots[1];
    critical_roots[5] = end;

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
    // Iterate over the intervals where roots may be found
    float begin_value = eval_poly(deriv_poly, begin);

    #pragma unroll
    for (int i = 2; i != 5; ++i) 
    {
        float root;
        if (poly5_has_root_newton_bisection(root, begin_value, deriv_poly, critical_roots[i], critical_roots[i+1], begin_value, tolerance))
        {
            critical_roots[i] = root;
        }
        else
        {
            // Create an empty interval for the next iteration
            critical_roots[i] = critical_roots[i-1];
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
    // Iterate over the intervals where roots may be found
    begin_value = eval_poly(deriv_poly, begin);

    #pragma unroll
    for (int i = 1; i != 5; ++i) 
    {
        // Try to find a root
        float root;
        if (poly5_has_root_newton_bisection(root, begin_value, deriv_poly, critical_roots[i], critical_roots[i+1], begin_value, tolerance))
        {
            critical_roots[i] = root;
        }
        else
        {
            // Create an empty interval for the next iteration
            critical_roots[i] = critical_roots[i-1];
        }
    }
    
    // degree = 5
    // Take the integral of the previous derivative (scaled such that the
    // constant coefficient can still be copied directly from poly)
    deriv_poly[5] = deriv_poly[4] * (1.0 / 5.0);
    deriv_poly[4] = deriv_poly[3] * (1.0 / 4.0);
    deriv_poly[3] = deriv_poly[2] * (1.0 / 3.0);
    deriv_poly[2] = deriv_poly[1] * (1.0 / 2.0);
    deriv_poly[1] = deriv_poly[0] * (1.0 / 1.0);
    deriv_poly[0] = poly[0];

    // Determine the value of this derivative at begin
    // Iterate over the intervals where sign change may be found
    begin_value = eval_poly(deriv_poly, begin);

    #pragma unroll
    for (int i = 0; i != 5; ++i) 
    {
        // Try to find sign change
        if (poly5_has_root_sign_change(begin_value, deriv_poly, critical_roots[i], critical_roots[i+1], begin_value))
        {
            return true;
        }
    }
 
    return false;
}

// Finds if the given polynomial has root in the interval [begin, end]
// using cubic deflation to quadratic
bool poly5_has_root_v2
(
    float poly[6], 
    float begin, 
    float end
){
    float tolerance = (end - begin) * MICRO_TOLERANCE;

    // Construct the quadratic derivative of the polynomial. We divide each
    // derivative by the factorial of its order = 3, such that the constant
    // coefficient can be copied directly from poly. That is a safeguard
    // against overflow and makes it easier to avoid spilling below. The
    // factors happen to be binomial coefficients then.

    // degree = 2
    float deriv_poly[6];
    deriv_poly[5] = 0.0;
    deriv_poly[4] = 0.0;
    deriv_poly[3] = 0.0;
    deriv_poly[2] = poly[5] * 10.0; 
    deriv_poly[1] = poly[4] * 4.0;  
    deriv_poly[0] = poly[3];        
    
    // Compute its two roots using the quadratic formula
    vec2 quad_roots = poly5_has_root_quadratic_roots(deriv_poly, begin, end, tolerance);

    // Construct the cubic derivative of the polynomial and solve
    // using the deflation method, using the previous quadratic roots
    // to locate a single cubic root, in the monotonic bracket

    // degree = 3
    // Take the integral of the previous derivative (scaled such that the
    // constant coefficient can still be copied directly from poly)
    deriv_poly[5] = 0.0;
    deriv_poly[4] = 0.0;
    deriv_poly[3] = deriv_poly[2] * (3.0 / 3.0);
    deriv_poly[2] = deriv_poly[1] * (3.0 / 2.0);
    deriv_poly[1] = deriv_poly[0] * (3.0 / 1.0);
    deriv_poly[0] = poly[2];

    // Compute its three roots using deflation of the cubic into quadratic
    vec3 cubic_roots = poly5_has_root_cubic_roots(deriv_poly, begin, end, quad_roots[0], quad_roots[1], tolerance);

    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched critical roots are set to begin
    float critical_roots[6];
    critical_roots[0] = begin;
    critical_roots[1] = begin;
    critical_roots[2] = cubic_roots[0];
    critical_roots[3] = cubic_roots[1];
    critical_roots[4] = cubic_roots[2];
    critical_roots[5] = end;

    // Work your way up to derivatives of higher degree until you reach the
    // polynomial itself. This implementation may seem peculiar: It always
    // treats the derivative as though it had degree 5 and it
    // constructs the derivatives in a contrived way. Changing that would
    // reduce the number of arithmetic instructions roughly by a factor of two.
    // However, it would also cause register spilling, which has a far more
    // negative impact on the overall run time. Profiling indicates that the
    // current implementation has no spilling whatsoever.    

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
    float begin_value = eval_poly(deriv_poly, begin);

    // Iterate over the intervals where roots may be found
    #pragma unroll
    for (int i = 1; i != 5; ++i) 
    {
        // Try to find a root
        float root;
        if (poly5_has_root_newton_bisection(root, begin_value, deriv_poly, critical_roots[i], critical_roots[i+1], begin_value, tolerance))
        {
            critical_roots[i] = root;
        }
        else
        {
            // Create an empty interval for the next iteration
            critical_roots[i] = critical_roots[i-1];
        }
    }
    
    // degree = 5
    // Take the integral of the previous derivative (scaled such that the
    // constant coefficient can still be copied directly from poly)
    deriv_poly[5] = deriv_poly[4] * (1.0 / 5.0);
    deriv_poly[4] = deriv_poly[3] * (1.0 / 4.0);
    deriv_poly[3] = deriv_poly[2] * (1.0 / 3.0);
    deriv_poly[2] = deriv_poly[1] * (1.0 / 2.0);
    deriv_poly[1] = deriv_poly[0] * (1.0 / 1.0);
    deriv_poly[0] = poly[0];

    // Determine the value of this derivative at begin
    begin_value = eval_poly(deriv_poly, begin);

    // Iterate over the intervals where sign change may be found
    #pragma unroll
    for (int i = 0; i != 5; ++i) 
    {
        // Try to find sign change
        if (poly5_has_root_sign_change(begin_value, deriv_poly, critical_roots[i], critical_roots[i+1], begin_value))
        {
            return true;
        }
    }
 
    return false;
}

#endif