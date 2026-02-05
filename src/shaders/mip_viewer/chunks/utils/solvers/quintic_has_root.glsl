
/* Soures
Finding Real Polynomial Roots on GPUs (https://momentsingraphics.de/GPUPolynomialRoots.html),
Shadertoy Spherical harmonics glyphs (https://www.shadertoy.com/view/dlGSDV),
Ray Tracing Spherical Harmonics Glyphs (https://momentsingraphics.de/VMV2023.html),
High-Performance Polynomial Solver Cem Yuksel (https://www.cemyuksel.com/research/polynomials/),
cyPolynomial.h class (https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h),
*/

#ifndef QUINTIC_HAS_ROOT
#define QUINTIC_HAS_ROOT

// How close we want to get in the real roots
#ifndef QUARTIC_ROOTS_TOLERANCE
#define QUARTIC_ROOTS_TOLERANCE 1e-6
#endif
// How to subdivide the samples in groups of 4 for parallel evaluation
#ifndef QUINTIC_SAMPLES_SUBDIVS
#define QUINTIC_SAMPLES_SUBDIVS 4
#endif


// Searches a single root of a quartic polynomial within a given interval.
// \param out_root The location of the found root.
// \param out_end_value The value of the given polynomial at end.
// \param poly Coefficients of the polynomial for which a root should be found.
//        Coefficient poly[i] is multiplied by x^i.
// \param begin The beginning of an interval where the polynomial is monotonic.
// \param end The end of said interval.
// \param begin_value The value of the given polynomial at begin.
// \param tolerance The error tolerance for the returned root location.
//        Typically the error will be much lower but in theory it can be
//        bigger.
// \return true if a root was found, false if no root exists.
bool quartic_root_bisection(
    out float out_root, 
    out float out_end_value,
    float poly[5], 
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
    out_end_value = poly[4];
    out_end_value = out_end_value * end + poly[3];
    out_end_value = out_end_value * end + poly[2];
    out_end_value = out_end_value * end + poly[1];
    out_end_value = out_end_value * end + poly[0];

    // If the values at both ends have the same non-zero sign, there is no root
    if (begin_value * out_end_value > 0.0) return false;

    // Otherwise, we find the root iteratively using Newton bisection (with
    // bounded iteration count)
    float current = 0.5 * (begin + end);

    #pragma no_unroll
    for (int i = 0; i < 20; ++i) 
    {
        // Evaluate the polynomial and its derivative
        float value = poly[4];
        value = value * current + poly[3];
        value = value * current + poly[2];
        value = value * current + poly[1];
        value = value * current + poly[0];

        // Shorten the interval
        bool right = begin_value * value > 0.0;
        begin = right ? current : begin;
        end = right ? end : current;

        // Apply Bisection method
        current = 0.5 * (begin + end);
    }

    out_root = current;
    return true;
}

// Searches a single root of a quartic polynomial within a given interval.
// \param out_root The location of the found root.
// \param out_end_value The value of the given polynomial at end.
// \param poly Coefficients of the polynomial for which a root should be found.
//        Coefficient poly[i] is multiplied by x^i.
// \param begin The beginning of an interval where the polynomial is monotonic.
// \param end The end of said interval.
// \param begin_value The value of the given polynomial at begin.
// \param tolerance The error tolerance for the returned root location.
//        Typically the error will be much lower but in theory it can be
//        bigger.
// \return true if a root was found, false if no root exists.
bool quartic_root_regula_falsi(
    out float out_root, 
    out float out_end_value,
    float poly[5], 
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
    float end_value = poly[4];
    end_value = end_value * end + poly[3];
    end_value = end_value * end + poly[2];
    end_value = end_value * end + poly[1];
    end_value = end_value * end + poly[0];
    out_end_value = end_value;

    // If the values at both ends have the same non-zero sign, there is no root
    if (begin_value * end_value > 0.0) return false;

    // Otherwise, we find the root iteratively using Neubauer method (with
    // bounded iteration count)   
    float current = 0.5 * (begin + end);

    // Remember which endpoint was replaced last time for  Illinois damping
    int last_replaced = -1; 

    #pragma no_unroll
    for (int i = 0; i < 10; ++i) 
    {
        // Evaluate the polynomial and its derivative
        float value = poly[4];
        value = value * current + poly[3];
        value = value * current + poly[2];
        value = value * current + poly[1];
        value = value * current + poly[0];

        // Shorten the interval
        bool right = begin_value * value > 0.0;
        begin = right ? current : begin;
        end = right ? end : current;
        begin_value = right ? value : begin_value;
        end_value = right ? end_value : value;

        // if the same endpoint was replaced twice in a row, halve the survivor's value.
        begin_value *= (last_replaced == 1 && !right) ? 0.5 : 1.0;
        end_value   *= (last_replaced == 0 &&  right) ? 0.5 : 1.0;
        last_replaced = right ? 0 : 1;

        // Compute differences
        float delta = begin - end;
        float delta_value = begin_value - end_value;    

        // Apply Neubauer step
        float step = (begin_value * delta) / delta_value;
        float next = begin - step;

        // Move along or stay the same
        current = abs(delta) < tolerance ? current : next;
    }

    out_root = current;
    return true;
}

// Searches a single root of a quartic polynomial within a given interval.
// \param out_root The location of the found root.
// \param out_end_value The value of the given polynomial at end.
// \param poly Coefficients of the polynomial for which a root should be found.
//        Coefficient poly[i] is multiplied by x^i.
// \param begin The beginning of an interval where the polynomial is monotonic.
// \param end The end of said interval.
// \param begin_value The value of the given polynomial at begin.
// \param tolerance The error tolerance for the returned root location.
//        Typically the error will be much lower but in theory it can be
//        bigger.
// \return true if a root was found, false if no root exists.
bool quartic_root_newton_bisection(
    out float out_root, 
    out float out_end_value,
    float poly[5], 
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
    out_end_value = poly[4];
    out_end_value = out_end_value * end + poly[3];
    out_end_value = out_end_value * end + poly[2];
    out_end_value = out_end_value * end + poly[1];
    out_end_value = out_end_value * end + poly[0];

    // If the values at both ends have the same non-zero sign, there is no root
    if (begin_value * out_end_value > 0.0) return false;

    // Otherwise, we find the root iteratively using Newton bisection (with
    // bounded iteration count)
    float current = 0.5 * (begin + end);

    #pragma no_unroll
    for (int i = 0; i < 10; ++i) 
    {
        // Evaluate the polynomial and its derivative
        float derivative = poly[4];
        float value = derivative * current + poly[3];
        derivative = derivative * current + value;
        value = value * current + poly[2];
        derivative = derivative * current + value;
        value = value * current + poly[1];
        derivative = derivative * current + value;
        value = value * current + poly[0];

        // Shorten the interval
        bool right = (begin_value * value > 0.0);
        begin = right ? current : begin;
        end = right ? end : current;

        // Apply Newton's method
        float step = value / derivative;
        float guess = current - step;

        // Pick a guess
        float middle = 0.5 * (begin + end);
        current = (guess >= begin && guess <= end) ? guess : middle;
    }

    out_root = current;
    return true;
}

// Searches a single root of a quintic polynomial within a given interval.
// \param out_end_value The value of the given polynomial at end.
// \param poly Coefficients of the polynomial for which a root should be found.
//        Coefficient poly[i] is multiplied by x^i.
// \param begin The beginning of an interval where the polynomial is monotonic.
// \param end The end of said interval.
// \param begin_value The value of the given polynomial at begin.
// \return true if a root was found, false if no root exists.
bool quintic_sign_change(
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
    return (begin_value * out_end_value <= 0.0);
}

// Finds if the given quintic has root polynomial in the interval [begin, end]
bool quintic_has_root(
    float poly[6], 
    float begin, 
    float end
){
    float tolerance = (end - begin) * QUARTIC_ROOTS_TOLERANCE;

    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched critical roots are set to begin
    float crit_roots[6];
    crit_roots[0] = begin;
    crit_roots[1] = begin;
    crit_roots[2] = begin;
    crit_roots[5] = end;

    // Construct the quadratic derivative of the polynomial. We divide each
    // derivative by the factorial of its order = 3, such that the constant
    // coefficient can be copied directly from poly. That is a safeguard
    // against overflow and makes it easier to avoid spilling below. The
    // factors happen to be binomial coefficients then.
    float deriv_poly[5];
    deriv_poly[4] = 0.0;
    deriv_poly[3] = 0.0;
    deriv_poly[2] = poly[5] * 10.0; 
    deriv_poly[1] = poly[4] * 4.0;  
    deriv_poly[0] = poly[3];        
    
    // Compute its two roots using the quadratic formula
    float discriminant = deriv_poly[1] * deriv_poly[1] - 4.0 * deriv_poly[0] * deriv_poly[2];
    if (discriminant >= 0.0) 
    {
        // Compute the quadratic roots using numerically stable solutions
        float sqrt_disc = sqrt(discriminant);
        float scaled_root = -0.5 * (deriv_poly[1] + sqrt_disc * sign(deriv_poly[1]));
        float root_0 = clamp(deriv_poly[0] / scaled_root, begin, end);
        float root_1 = clamp(scaled_root / deriv_poly[2], begin, end); 

        crit_roots[3] = min(root_0, root_1);
        crit_roots[4] = max(root_0, root_1);
    }
    else
    {
        crit_roots[3] = begin;
        crit_roots[4] = begin;
    }

    // Work your way up to derivatives of higher degree until you reach the
    // polynomial itself. This implementation may seem peculiar: It always
    // treats the derivative as though it had degree 5 and it
    // constructs the derivatives in a contrived way. Changing that would
    // reduce the number of arithmetic instructions roughly by a factor of two.
    // However, it would also cause register spilling, which has a far more
    // negative impact on the overall run time. Profiling indicates that the
    // current implementation has no spilling whatsoever.    
  
    // degrees = 3,4,5
    #pragma no_unroll
    for (int degree = 3; degree <= 4; ++degree) 
    {
        // Take the integral of the previous derivative (scaled such that the
        // constant coefficient can still be copied directly from poly)
        float prev_derivative_order = float(6 - degree);
        deriv_poly[4] = deriv_poly[3] * (prev_derivative_order * (1.0 / 4.0));
        deriv_poly[3] = deriv_poly[2] * (prev_derivative_order * (1.0 / 3.0));
        deriv_poly[2] = deriv_poly[1] * (prev_derivative_order * (1.0 / 2.0));
        deriv_poly[1] = deriv_poly[0] * (prev_derivative_order * (1.0 / 1.0));
     
        // Copy the constant coefficient without causing spilling. This part
        // would be harder if the derivative were not scaled the way it is.
        deriv_poly[0] = (degree == 4) ? poly[1] : deriv_poly[0];
        deriv_poly[0] = (degree == 3) ? poly[2] : deriv_poly[0];

        // Determine the value of this derivative at begin
        float begin_value = deriv_poly[4];
        begin_value = begin_value * begin + deriv_poly[3];
        begin_value = begin_value * begin + deriv_poly[2];
        begin_value = begin_value * begin + deriv_poly[1];
        begin_value = begin_value * begin + deriv_poly[0];

        // Iterate over the intervals where roots may be found
        #pragma unroll
        for (int i = 0; i <= 4; ++i) 
        {
            if (i < 5 - degree)  continue;

            float current_begin = crit_roots[i];
            float current_end = crit_roots[i + 1];

            // Try to find a root
            float current_root;
            if (quartic_root_bisection(current_root, begin_value, deriv_poly, current_begin, current_end, begin_value))
            {
                crit_roots[i] = current_root;
            }
            else
            {
                // Create an empty interval for the next iteration
                crit_roots[i] = crit_roots[i - 1];
            }
        }
    }

    // Determine the value of this derivative at begin
    float begin_value = poly[5];
    begin_value = begin_value * begin + poly[4];
    begin_value = begin_value * begin + poly[3];
    begin_value = begin_value * begin + poly[2];
    begin_value = begin_value * begin + poly[1];
    begin_value = begin_value * begin + poly[0];

    // Iterate over the intervals where roots may be found
    #pragma unroll
    for (int i = 0; i < 5; ++i) 
    {
        float current_begin = crit_roots[i];
        float current_end = crit_roots[i + 1];

        if (quintic_sign_change(begin_value, poly, current_begin, current_end, begin_value))
        {
            return true;
        }
    }
    
    return false;
}

// Finds if the given quintic has root polynomial in the interval [begin, end]
bool quintic_has_root_deflate(
    float poly[6], 
    float begin, 
    float end
){
    float tolerance = (end - begin) * QUARTIC_ROOTS_TOLERANCE;

    // The last entry in the root array is set to end to make it easier to
    // iterate over relevant intervals, all untouched critical roots are set to begin
    float crit_roots[6];
    crit_roots[0] = begin;
    crit_roots[1] = begin;
    crit_roots[2] = begin;
    crit_roots[5] = end;

    // Construct the quadratic derivative of the polynomial. We divide each
    // derivative by the factorial of its order = 3, such that the constant
    // coefficient can be copied directly from poly. That is a safeguard
    // against overflow and makes it easier to avoid spilling below. The
    // factors happen to be binomial coefficients then.
    float deriv_poly[5];
    deriv_poly[4] = 0.0;
    deriv_poly[3] = 0.0;
    deriv_poly[2] = poly[5] * 10.0; 
    deriv_poly[1] = poly[4] * 4.0;  
    deriv_poly[0] = poly[3];        
    
    // Compute its two roots using the quadratic formula
    float discriminant = deriv_poly[1] * deriv_poly[1] - 4.0 * deriv_poly[0] * deriv_poly[2];
    if (discriminant >= 0.0) 
    {
        // Compute the quadratic roots using numerically stable solutions
        float sqrt_disc = sqrt(discriminant);
        float scaled_root = -0.5 * (deriv_poly[1] + sqrt_disc * sign(deriv_poly[1]));
        float root_0 = clamp(deriv_poly[0] / scaled_root, begin, end);
        float root_1 = clamp(scaled_root / deriv_poly[2], begin, end); 

        crit_roots[3] = min(root_0, root_1);
        crit_roots[4] = max(root_0, root_1);
    }
    else
    {
        crit_roots[3] = begin;
        crit_roots[4] = begin;
    }

    // Work your way up to derivatives of higher degree until you reach the
    // polynomial itself. This implementation may seem peculiar: It always
    // treats the derivative as though it had degree 5 and it
    // constructs the derivatives in a contrived way. Changing that would
    // reduce the number of arithmetic instructions roughly by a factor of two.
    // However, it would also cause register spilling, which has a far more
    // negative impact on the overall run time. Profiling indicates that the
    // current implementation has no spilling whatsoever.   

    #pragma no_unroll 
    for (int degree = 3; degree <= 4; ++degree) 
    {
        // Take the integral of the previous derivative (scaled such that the
        // constant coefficient can still be copied directly from poly)
        float prev_derivative_order = float(6 - degree);
        deriv_poly[4] = deriv_poly[3] * (prev_derivative_order * (1.0 / 4.0));
        deriv_poly[3] = deriv_poly[2] * (prev_derivative_order * (1.0 / 3.0));
        deriv_poly[2] = deriv_poly[1] * (prev_derivative_order * (1.0 / 2.0));
        deriv_poly[1] = deriv_poly[0] * (prev_derivative_order * (1.0 / 1.0));
     
        // Copy the constant coefficient without causing spilling. This part
        // would be harder if the derivative were not scaled the way it is.
        deriv_poly[0] = (degree == 4) ? poly[1] : deriv_poly[0];
        deriv_poly[0] = (degree == 3) ? poly[2] : deriv_poly[0];

        // Determine the value of this derivative at begin
        float begin_value = deriv_poly[4];
        begin_value = begin_value * begin + deriv_poly[3];
        begin_value = begin_value * begin + deriv_poly[2];
        begin_value = begin_value * begin + deriv_poly[1];
        begin_value = begin_value * begin + deriv_poly[0];

        // Start deflated polynomial as the derivative
        float defl_poly[5] = deriv_poly;

        // Iterate over the intervals where roots may be found
        float current_root = begin; 
        int num_roots = 0; 
        
        #pragma unroll
        for (int i = 0; i < 5; ++i) 
        {
            if (i < 5 - degree)  continue;

            float current_begin = crit_roots[i];
            float current_end = crit_roots[i + 1];

            // Try to find a root
            if (quartic_root_bisection(current_root, begin_value, deriv_poly, current_begin, current_end, begin_value))
            {
                crit_roots[i] = current_root; 
                if (i < 4) num_roots++;

                float d4 = 0.0;        
                float d3 = d4 * current_root + defl_poly[4];
                float d2 = d3 * current_root + defl_poly[3];
                float d1 = d2 * current_root + defl_poly[2];
                float d0 = d1 * current_root + defl_poly[1];
                // float res = d0 * current_root + defl_poly[0];

                defl_poly[4] = d4;
                defl_poly[3] = d3;
                defl_poly[2] = d2;
                defl_poly[1] = d1;
                defl_poly[0] = d0;
            }
            else
            {
                // Create an empty interval for the next iteration
                crit_roots[i] = crit_roots[i - 1];
            }

            // When the deflated polynomial becomes a quadratic
            // break the loop and solve for the remaining roots
            if (num_roots == degree - 2) break;
        }

        // Compute quadratic roots in [current_root, end]
        // if roots where found clamp them to bracket to keep the order
        if (num_roots == degree - 2) 
        {          
            // Compute its two roots using the stable quadratic formula
            float discriminant = defl_poly[1] * defl_poly[1] - 4.0 * defl_poly[0] * defl_poly[2];
            if (discriminant >= 0.0) 
            {
                // Compute the quadratic roots using numerically stable solutions
                float sqrt_disc = sqrt(discriminant);
                float scaled_root = -0.5 * (defl_poly[1] + sqrt_disc * sign(defl_poly[1]));
                float root_0 = clamp(defl_poly[0] / scaled_root, begin, end);
                float root_1 = clamp(scaled_root / defl_poly[2], begin, end); 

                crit_roots[3] = min(root_0, root_1);
                crit_roots[4] = max(root_0, root_1);
            }
            else
            {
                crit_roots[3] = begin;
                crit_roots[4] = begin;
            }
        }
    }

    // Determine the value of this derivative at begin
    float begin_value = poly[5];
    begin_value = begin_value * begin + poly[4];
    begin_value = begin_value * begin + poly[3];
    begin_value = begin_value * begin + poly[2];
    begin_value = begin_value * begin + poly[1];
    begin_value = begin_value * begin + poly[0];

    // Iterate over the intervals where roots may be found
    #pragma unroll
    for (int i = 0; i < 5; ++i) 
    {
        float current_begin = crit_roots[i];
        float current_end = crit_roots[i + 1];

        if (quintic_sign_change(begin_value, poly, current_begin, current_end, begin_value))
        {
            return true;
        }
    }
    
    return false;
}

// Finds if the given quintic has root polynomial in the interval [begin, end]
// via uniform sampling in 4-wide tiles (SIMD). Checks sign changes.
bool quintic_has_root_sample(
    float poly[6], 
    const float begin, 
    const float end
){
    float delta = (end - begin) / float(QUINTIC_SAMPLES_SUBDIVS * 4);
    float step = delta * 4.0;

    // Start previous value at begin
    float prev = ((((poly[5]*begin + poly[4])*begin + poly[3])*begin + poly[2])*begin + poly[1])*begin + poly[0];

    // First tile positions at begin
    vec4 pos = begin + delta * vec4(1.0, 2.0, 3.0, 4.0);

    #pragma unroll
    for (int i = 0; i < QUINTIC_SAMPLES_SUBDIVS; ++i) 
    {
        // Horner on 4 positions at once
        vec4 v = vec4(poly[5]);
        v = v * pos + poly[4];
        v = v * pos + poly[3];
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
