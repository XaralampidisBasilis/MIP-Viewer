
/* Soures
Finding Real Polynomial Roots on GPUs (https://momentsingraphics.de/GPUPolynomialRoots.html),
Shadertoy Spherical harmonics glyphs (https://www.shadertoy.com/view/dlGSDV),
Ray Tracing Spherical Harmonics Glyphs (https://momentsingraphics.de/VMV2023.html),
High-Performance Polynomial Solver Cem Yuksel (https://www.cemyuksel.com/research/polynomials/),
cyPolynomial.h class (https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h),
Based on Blinn's paper (https://courses.cs.washington.edu/courses/cse590b/13au/lecture_notes/solvecubic_p5.pdf),
Article by Christoph Peters (https://momentsingraphics.de/CubicRoots.html#_Blinn07b),
Shadertoy Cubic Equation Solver II (https://www.shadertoy.com/view/7tBGzK),
Shadertoy Quartic Reflections https://www.shadertoy.com/view/flBfzm,
*/

#ifndef CUBIC_ROOTS
#define CUBIC_ROOTS

// When there are fewer intersections/roots than theoretically possible, some
// array entries are set to this value
#ifndef CUBIC_NO_INTERSECTION
#define CUBIC_NO_INTERSECTION 3.4e38
#endif
// How close we want to get in the real roots
#ifndef CUBIC_ROOTS_TOLERANCE
#define CUBIC_ROOTS_TOLERANCE 1e-6
#endif
// The number of newton bisection iterations to reach 
// the desired error tolerance
#ifndef CUBIC_NEWTON_BISECTION_ITERS
#define CUBIC_NEWTON_BISECTION_ITERS 6
#endif

#ifndef EVAL_POLY
#include "../math/eval_poly"
#endif
#ifndef CBRT
#include "../math/cbrt"
#endif
#ifndef SQRT_3
#define SQRT_3 1.73205080757
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
bool cubic_roots_newton_bisection
(
    out float out_root, 
    out float out_end_value,
    vec4 poly, 
    float begin, 
    float end,
    float begin_value, 
    float tolerance
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

    // If the values at both ends have the same non-zero sign, there is no root
    if (begin_value * out_end_value > 0.0) return false;

    // Otherwise, we find the root iteratively using Newton bisection (with
    // bounded iteration count)
    float current = 0.5 * (begin + end);

    #pragma no_unroll
    for (int i = 0; i < CUBIC_NEWTON_BISECTION_ITERS; ++i) 
    {
        // Evaluate the polynomial and its derivative
        float derivative = poly[3];
        float value = poly[3] * current + poly[2];
        #pragma unroll
        for (int j = 1; j >= 0; --j) 
        {
            derivative = derivative * current + value;
            value = value * current + poly[j];
        }
    
        // Shorten the interval
        bool right = begin_value * value > 0.0;
        begin = right ? current : begin;
        end = right ? end : current;

        // Apply Newton's method
        float step = value / derivative;
        float guess = current - step;

        // Pick a guess
        float middle = 0.5 * (begin + end);
        float next = (guess >= begin && guess <= end) ? guess : middle;

        // Move along or terminate
        bool done = abs(step) < tolerance;
        current = next;
        if (done) break;
    }

    out_root = current;
    return true;
}

// Finds all roots of the cubic polynomial in the interval [begin, end] and
// writes them to out_roots. Some entries will be CUBIC_NO_INTERSECTION but other 
// than that the array is sorted. The last entry is always CUBIC_NO_INTERSECTION.
void cubic_roots(
    out vec4 out_roots, 
    vec4 poly, 
    float begin, 
    float end
){
    float tolerance = (end - begin) * CUBIC_ROOTS_TOLERANCE;

    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched roots are set to begin
    out_roots[0] = begin;
    out_roots[3] = end;

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

        out_roots[1] = min(root_0, root_1);
        out_roots[2] = max(root_0, root_1);
    }
    else
    {
        out_roots[1] = begin;
        out_roots[2] = begin;
    }

    // Take the integral of the previous derivative (scaled such that the
    // constant coefficient can still be copied directly from poly)
    // Copy the constant coefficient without causing spilling. This part
    // would be harder if the derivative were not scaled the way it is.
    deriv_poly[3] = deriv_poly[2] * (1.0 / 3.0);
    deriv_poly[2] = deriv_poly[1] * (1.0 / 2.0);
    deriv_poly[1] = deriv_poly[0] * (1.0 / 1.0);
    deriv_poly[0] = poly[0];

    // Determine the value of this derivative at begin
    float begin_value = deriv_poly[3];
    begin_value = begin_value * begin + deriv_poly[2];
    begin_value = begin_value * begin + deriv_poly[1];
    begin_value = begin_value * begin + deriv_poly[0];

    // Iterate over the intervals where roots may be found
    #pragma unroll
    for (int i = 0; i <= 2; ++i) 
    {
        float current_begin = out_roots[i];
        float current_end = out_roots[i + 1];

        // Try to find a root
        float current_root;
        if (cubic_roots_newton_bisection(current_root, begin_value, deriv_poly, current_begin, current_end, begin_value, tolerance))
        {
            out_roots[i] = current_root;
        }
        else
        {
            out_roots[i] = CUBIC_NO_INTERSECTION;
        }
    }
 
    // We no longer need this array entry
    out_roots[3] = CUBIC_NO_INTERSECTION;
}

// Finds all roots of the given quintic polynomial in the interval [begin, end] using cubic deflation and
// writes them to out_roots using the cubic deflation method. Some entries will be QUINTIC_NO_INTERSECTION but other 
// than that the array is sorted. The last entry is always QUINTIC_NO_INTERSECTION.
void cubic_roots_deflate(
    out vec4 out_roots, 
    vec4 poly, 
    float begin, 
    float end
){
    float tolerance = (end - begin) * CUBIC_ROOTS_TOLERANCE;

    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched roots are set to begin
    out_roots[0] = begin;
    out_roots[3] = end;

    // Construct the quadratic derivative of the polynomial. We divide each
    // derivative by the factorial of its order, such that the constant
    // coefficient can be copied directly from poly. That is a safeguard
    // against overflow and makes it easier to avoid spilling below. The
    // factors happen to be binomial coefficients then.
    vec4 deriv_poly;
    deriv_poly[3] = 0.0;
    deriv_poly[2] = poly[3] * 3.0;
    deriv_poly[1] = poly[2] * 2.0;
    deriv_poly[0] = poly[1];

    // Compute its two roots using the quadratic formula
    float discriminant = deriv_poly[1] * deriv_poly[1] - 4.0 * deriv_poly[0] * deriv_poly[2];
    if (discriminant >= 0.0) 
    {
        // Compute the quadratic roots using numerically stable solutions
        float sqrt_disc = sqrt(discriminant);
        float scaled_root = -0.5 * (deriv_poly[1] + sqrt_disc * sign(deriv_poly[1]));
        float root_0 = clamp(deriv_poly[0] / scaled_root, begin, end);
        float root_1 = clamp(scaled_root / deriv_poly[2], begin, end); 

        out_roots[1] = min(root_0, root_1);
        out_roots[2] = max(root_0, root_1);
    }
    else
    {
        out_roots[1] = begin;
        out_roots[2] = begin;
    }

    // Take the integral of the previous derivative (scaled such that the
    // constant coefficient can still be copied directly from poly)
    // Copy the constant coefficient without causing spilling. This part
    // would be harder if the derivative were not scaled the way it is.
    deriv_poly[3] = deriv_poly[2] * (1.0 / 3.0);
    deriv_poly[2] = deriv_poly[1] * (1.0 / 2.0);
    deriv_poly[1] = deriv_poly[0] * (1.0 / 1.0);
    deriv_poly[0] = poly[0];

    // Determine the value of this derivative at begin
    float begin_value = deriv_poly[3];
    begin_value = begin_value * begin + deriv_poly[2];
    begin_value = begin_value * begin + deriv_poly[1];
    begin_value = begin_value * begin + deriv_poly[0];

    // Iterate over the intervals where roots may be found
    bool solve_quadratic = false;
    float current_root = begin;

    #pragma unroll
    for (int i = 0; i <= 2; ++i) 
    {
        if (solve_quadratic) continue;

        float current_begin = out_roots[i];
        float current_end = out_roots[i + 1];

        // Try to find a root
        if (cubic_roots_newton_bisection(current_root, begin_value, deriv_poly, current_begin, current_end, begin_value, tolerance))
        {
            out_roots[i] = current_root;

            // If we found a root but we are in the last bracket deflation is not needed
            solve_quadratic = (i != 2);
        }
        else
        {
            out_roots[i] = CUBIC_NO_INTERSECTION;
        }
    }

    // Compute quadratic roots in [current_root, end]
    if (solve_quadratic) 
    {
        // deflate the cubic to quadratic
        deriv_poly[2] = deriv_poly[2] + deriv_poly[3] * current_root; 
        deriv_poly[1] = deriv_poly[1] + deriv_poly[2] * current_root; 

        // If quadratic discriminant is negative there are no roots
        float discriminant = deriv_poly[2] * deriv_poly[2] - 4.0 * deriv_poly[1] * deriv_poly[3];
        if (discriminant >= 0.0) 
        {
            // Compute the quadratic roots using numerically stable solutions
            float sqrt_disc = sqrt(discriminant);
            float scaled_root = -0.5 * (deriv_poly[2] + sqrt_disc * sign(deriv_poly[2]));
            float root_0 = clamp(deriv_poly[1] / scaled_root, current_root, end);
            float root_1 = clamp(scaled_root / deriv_poly[3], current_root, end); 

            out_roots[1] = min(root_0, root_1);
            out_roots[2] = max(root_0, root_1);
        }
        else
        {
            out_roots[1] = CUBIC_NO_INTERSECTION;
            out_roots[2] = CUBIC_NO_INTERSECTION;
        }
    }

    // We no longer need this array entry
    out_roots[3] = CUBIC_NO_INTERSECTION;
}

// Finds all roots of the cubic polynomial
// and writes them to out_roots
vec3 cubic_roots_analytic(in vec4 poly)
{
    // Flip to minimize instability
    bool flip = abs(poly.z * poly.x) >= abs(poly.y * poly.w);
    vec4 flip_poly = flip ? poly.wzyx : poly;

    // Normalize to simplify computations
    vec3 norm_poly = flip_poly.xyz / flip_poly.w;
    norm_poly.yz /= 3.0;

    // Compute hessian coefficients eq(0.4)
    vec3 hessian = vec3(
        norm_poly.y - norm_poly.z * norm_poly.z,                          // δ1 = c.w * c.y - c.z^2
        norm_poly.x - norm_poly.y * norm_poly.z,                          // δ2 = c.w * c.x - c.y * c.z
        dot(vec2(norm_poly.z, -norm_poly.y), norm_poly.xy)    // δ3 = c.z * c.x - c.y^2
    );
    hessian.y /= 2.0;

    // compute cubic discriminant eq(0.7)
    float disc = dot(vec2(hessian.x, -hessian.y), hessian.zy); // Δ = δ1 * δ3 - δ2^2
    float sqrt_disc = sqrt(abs(disc));

    // compute depressed cubic eq(0.16), r[0] + r[1] * x + x^3 eq(0.11) eq(0.16)
    vec2 reduced_poly = vec2(hessian.y - hessian.x * norm_poly.z, hessian.x);
        
    // compute real root using cubic root formula for one real and two complex roots eq(0.15)
    float root1 = cbrt(-reduced_poly.x + sqrt_disc) + cbrt(-reduced_poly.x - sqrt_disc);

    // compute cubic roots using complex number formula eq(0.14)  
    float theta = atan(sqrt_disc, -reduced_poly.x) / 3.0;
    mat2 proj = mat2(-1.0, -SQRT_3, -1.0, SQRT_3);

    // compute three roots via rotation, applying complex root formula eq(0.14)
    vec2 scaled_roots = vec2(cos(theta), sin(theta));
    vec3 roots3 = vec3(scaled_roots * proj, scaled_roots.x * 2.0);

    // Revert transformation eq(0.2) and eq(0.16)
    roots3 *= sqrt(max(-reduced_poly.y, 0.0)); 

    // Revert transformations for selected roots
    vec3 roots = (disc > 0.0) ? roots3 : vec3(root1);
    roots = roots - norm_poly.z;
    roots = flip ? 1.0 / roots : roots;

    // Improve numerical stability of roots with Newton–Raphson correction
    vec3 values, derivative;
    #pragma unroll
    for (int i = 0; i < 2; ++i) 
    {
        values = eval_poly(poly, roots, derivative);
        roots -= values / derivative; 
    }

    // return result
    return roots;
}

#endif