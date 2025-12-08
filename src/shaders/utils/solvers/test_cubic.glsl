#ifndef TEST_CUBIC
#define TEST_CUBIC

#ifndef CUBIC_ROOTS
#include "./cubic_roots"
#endif
#ifndef IS_CUBIC_SOLVABLE
#include "./is_cubic_solvable"
#endif
#ifndef SORT
#include "../math/sort"
#endif
#ifndef SORT
#include "../math/sort"
#endif
#ifndef MILLI_TOLERANCE
#define MILLI_TOLERANCE 1e-3
#endif
#ifndef MICRO_TOLERANCE
#define MICRO_TOLERANCE 1e-6
#endif

// test functions
bvec3 test_cubic_residue(in vec4 c, in float x0)
{
    // Get roots from your solver
    vec3 x0_x1_x2 = cubic_roots(c);

    // Evaluate the original polynomial at each root
    vec3 y0_y1_y2 = eval_poly(c, x0_x1_x2);

    // Check if values are near zero
    bvec3 b0_b1_b2 = lessThan(abs(y0_y1_y2), vec3(MILLI_TOLERANCE));

    // Check if any value is near zero
    return b0_b1_b2;
}

bvec3 test_cubic_roots(in vec4 c, in float x0, in vec3 r0_r1_r2)
{
    // Get roots from your solver
    vec3 x0_x1_x2 = cubic_roots(c);

    // Sort roots in ascending form
    sort(x0_x1_x2);
    sort(r0_r1_r2);

    // Check if roots are approximated well
    bvec3 b0_b1_b2 = lessThan(abs(x0_x1_x2 - r0_r1_r2), vec3(MILLI_TOLERANCE));

    // Return true if all roots are approximated well
    return b0_b1_b2;
}

// test cases
bool test_cubic1_residue(out bvec3 test[7])
{
    vec4 c;

    // Case 0: (x + 1)(x^2 + 1) = x^3 + x^2 + x + 1 → real root at -1
    c = vec4(1.0, 1.0, 1.0, 1.0);
    test[0] = test_cubic_residue(c, 0.0);

    // Case 1: (x - 2)(x^2 + 4) = x^3 - 2x^2 + 4x - 8 → real root at 2
    c = vec4(-8.0, 4.0, -2.0, 1.0);
    test[1] = test_cubic_residue(c, 0.0);

    // Case 2: (x + 3)(x^2 + x + 1) = x^3 + 4x^2 + 4x + 3 → real root at -3
    c = vec4(3.0, 4.0, 4.0, 1.0);
    test[2] = test_cubic_residue(c, 0.0);

    // Case 3: (x - 0.25)(x^2 + 2x + 5) = x^3 + 1.5x^2 + 4.5x - 1.25
    c = vec4(-1.25, 4.5, 1.75, 1.0);
    test[3] = test_cubic_residue(c, 0.0);

    // Case 4: (x + 1.5)(x^2 - x + 2) = x^3 + 0.5x^2 + 0.5x + 3.0
    c = vec4(3.0, 0.5, 0.5, 1.0);
    test[4] = test_cubic_residue(c, 0.0);

    // Case 5: (x - 0.00001)(x^2 + 1) = x^3 + x^2 + x - 0.00001
    c = vec4(-0.00001, 1.0, -0.00001, 1.0);
    test[5] = test_cubic_residue(c, -1.0);

    // Case 6: (x + 5)(x^2 - 2x + 10) = x^3 + 3x^2 + 0x - 50
    c = vec4(50.0, 0.0, 3.0, 1.0);
    test[6] = test_cubic_residue(c, 0.0);

    // Combine all test cases
    return all(test[0]) && all(test[1]) && all(test[2]) && all(test[3]) && all(test[4]) && all(test[5]) && all(test[6]);
}

bool test_cubic3_residue(out bvec3 test[7])
{
    vec4 c;

    // Case 0: (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
    c = vec4(-6.0, 11.0, -6.0, 1.0);
    test[0] = test_cubic_residue(c, 0.0);

    // Case 1: (x+2)(x)(x-2) = x^3 - 4x
    c = vec4(0.0, -4.0, 0.0, 1.0);
    test[1] = test_cubic_residue(c, 1.0);

    // Case 2: (x+1)(x-1)(x-3) = x^3 - 3x^2 - x + 3
    c = vec4(3.0, -1.0, -3.0, 1.0);
    test[2] = test_cubic_residue(c, 0.0);

    // Case 3: (x-0.5)(x-1.5)(x-3.0) = x^3 - 5x^2 + 6.75x - 2.25
    c = vec4(-2.25, 6.75, -5.0, 1.0);
    test[3] = test_cubic_residue(c, 0.0);

    // Case 4: (x + 1.5)(x - 0.5)(x - 3.5) = x^3 - 2.5x^2 - 4.25x + 2.625
    c = vec4(2.625, -4.25, -2.5, 1.0);
    test[4] = test_cubic_residue(c, 0.0);

    // Case 5: (x - 1.00001)(x - 1)(x - 2) → numerically close roots
    c = vec4(-2.00002, 5.00003, -4.00001, 1.0);
    test[5] = test_cubic_residue(c, 0.0);
    
    // Case 6: (x - 1.00001)(x - 0.99999)(x - 1) → numerically close roots
    c = vec4(-0.9999999999 , 2.9999999999, -3.0, 1.0);
    test[6] = test_cubic_residue(c, 0.0);

    // Combine all test cases
    return all(test[0]) && all(test[1]) && all(test[2]) && all(test[3]) && all(test[4]) && all(test[5]) && all(test[6]);
}

bool test_cubic1_roots(out bvec3 test[7])
{
    vec3 r; vec4 c;

    // Case 0: (x + 1)(x^2 + 1) = x^3 + x^2 + x + 1 → real root at -1
    r = vec3(-1.0, 0.0, 0.0); // Only real root expected, other slots are placeholders
    c = vec4(1.0, 1.0, 1.0, 1.0);
    test[0] = test_cubic_roots(c, 0.0, r);

    // Case 1: (x - 2)(x^2 + 4) = x^3 - 2x^2 + 4x - 8 → real root at 2
    r = vec3(2.0, 0.0, 0.0);
    c = vec4(-8.0, 4.0, -2.0, 1.0);
    test[1] = test_cubic_roots(c, 0.0, r);

    // Case 2: (x + 3)(x^2 + x + 1) = x^3 + 4x^2 + 4x + 3 → real root at -3
    r = vec3(-3.0, 0.0, 0.0);
    c = vec4(3.0, 4.0, 4.0, 1.0);
    test[2] = test_cubic_roots(c, 0.0, r);

    // Case 3: (x - 0.25)(x^2 + 2x + 5) = x^3 + 1.5x^2 + 4.5x - 1.25
    r = vec3(0.25, 0.0, 0.0);
    c = vec4(-1.25, 4.5, 1.75, 1.0);
    test[3] = test_cubic_roots(c, 0.0, r);

    // Case 4: (x + 1.5)(x^2 - x + 2) = x^3 + 0.5x^2 + 0.5x + 3.0
    r = vec3(-1.5, 0.0, 0.0);
    c = vec4(3.0, 0.5, 0.5, 1.0);
    test[4] = test_cubic_roots(c, 0.0, r);

    // Case 5: (x - 0.00001)(x^2 + 1) = x^3 + x^2 + x - 0.00001
    r = vec3(0.00001, 1.0, 1.0);
    c = vec4(-0.00001, 1.0, -0.00001, 1.0);
    test[5] = test_cubic_roots(c, 1.0, r);

    // Case 6: (x + 5)(x^2 - 2x + 10) = x^3 + 3x^2 + 0x - 50
    r = vec3(-5.0, 0.0, 0.0);
    c = vec4(-50.0, 0.0, 3.0, 1.0);
    test[6] = test_cubic_roots(c, 0.0, r);

    // Combine all test cases
    return all(test[0]) && all(test[1]) && all(test[2]) && all(test[3]) && all(test[4]) && all(test[5]) && all(test[6]);
}

bool test_cubic3_roots(out bvec3 test[7])
{
    vec3 r; vec4 c;

    // Case 0: (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
    r = vec3(1.0, 2.0, 3.0);
    c = vec4(-6.0, 11.0, -6.0, 1.0);
    test[0] = test_cubic_roots(c, 0.0, r);

    // Case 1: (x+2)(x)(x-2) = x^3 - 4x
    r = vec3(-2.0, 0.0, 2.0);
    c = vec4(0.0, -4.0, 0.0, 1.0);
    test[1] = test_cubic_roots(c, 1.0, r);

    // Case 2: (x+1)(x-1)(x-3) = x^3 - 3x^2 - x + 3
    r = vec3(-1.0, 1.0, 3.0);
    c = vec4(3.0, -1.0, -3.0, 1.0);
    test[2] = test_cubic_roots(c, 0.0, r);

    // Case 3: (x-0.5)(x-1.5)(x-3.0) = x^3 - 5x^2 + 6.75x - 2.25
    r = vec3(0.5, 1.5, 3.0);
    c = vec4(-2.25, 6.75, -5.0, 1.0);
    test[3] = test_cubic_roots(c, 0.0, r);

    // Case 4: (x + 1.5)(x - 0.5)(x - 3.5) = x^3 - 2.5x^2 - 4.25x + 2.625
    r = vec3(-1.5, 0.5, 3.5);
    c = vec4(2.625, -4.25, -2.5, 1.0);
    test[4] = test_cubic_roots(c, 0.0, r);

    // Case 5: (x - 1.00001)(x - 1)(x - 2) → numerically close roots
    r = vec3(1.0, 1.00001, 2.0);
    c = vec4(-2.00002, 5.00003, -4.00001, 1.0);
    test[5] = test_cubic_roots(c, 0.0, r);
    
   // Case 6: (x - 1.00001)(x - 0.99999)(x - 1) → numerically close roots
    r = vec3(0.99999, 1.00001, 1.0);
    c = vec4(-0.9999999999 , 2.9999999999, -3.0, 1.0);
    test[6] = test_cubic_roots(c, 0.0, r);

    // Combine all test cases
    return all(test[0]) && all(test[1]) && all(test[2]) && all(test[3]) && all(test[4]) && all(test[5]) && all(test[6]);
}


// bool test_cubic_1(out bool test[6])
// {
//     vec4 c;

//     // Case 0: (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6 → three real roots
//     c = vec4(-6.0, 11.0, -6.0, 1.0);
//     test[0] = check_cubic_residue(c, 0.0);

//     // Case 1: (x-2)^3 = x^3 - 6x^2 + 12x - 8 → triple root
//     c = vec4(-8.0, 12.0, -6.0, 1.0);
//     test[1] = check_cubic_residue(c, 0.0);

//     // Case 2: (x-1)^2(x+2) = x^3 - 3x^2 + 0x + 2 → double real root, single real root
//     c = vec4(2.0, 0.0, -3.0, 1.0);
//     test[2] = check_cubic_residue(c, 0.0);

//     // Case 3: x^3 = 0 → triple root at 0
//     c = vec4(0.0, 0.0, 0.0, 1.0);
//     test[3] = check_cubic_residue(c, 1.0);

//     // Case 4: (x + 1)(x^2 + 4) = x^3 + x^2 + 4x + 4 → one real root, two complex
//     c = vec4(4.0, 4.0, 1.0, 1.0);
//     test[4] = check_cubic_residue(c, 0.0); 

//     // Case 5: ill-conditioned roots: (x-1)(x-1.00001)(x-2)
//     c = vec4(-2.00002 , 5.00003, -4.00001, 1.0);
//     test[5] = check_cubic_residue(c, 0.0);

//     return test[0] && test[1] && test[2] && test[3] && test[4] && test[5];
// }

// bool test_cubic_2(out bool test[6])
// {
//     vec2 xa_xb; vec4 c;

//     // Case 0: (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6 → three real roots
//     xa_xb = vec2(0.0, 4.0);
//     c = vec4(-6.0, 11.0, -6.0, 1.0);
//     test[0] = is_cubic_solvable(c, 0.0, xa_xb);

//     // Case 1: (x-2)^3 = x^3 - 6x^2 + 12x - 8 → triple root
//     xa_xb =  vec2(1.0, 3.0);
//     c = vec4(-8.0, 12.0, -6.0, 1.0);
//     test[1] = is_cubic_solvable(c, 0.0, xa_xb);

//     // Case 2: (x-1)^2(x+2) = x^3 - 3x^2 + 0x + 2 → double real root, single real root
//     xa_xb =  vec2(1.0, 3.0);
//     c = vec4(2.0, 0.0, -3.0, 1.0);
//     test[2] = is_cubic_solvable(c, 0.0, xa_xb);

//     // Case 3: x^3 = 0 → triple root at 0
//     xa_xb =  vec2(-1.0, 1.0);
//     c = vec4(0.0, 0.0, 0.0, 1.0);
//     test[3] = is_cubic_solvable(c, 0.0, xa_xb);

//     // Case 4: (x + 1)(x^2 + 4) = x^3 + x^2 + 4x + 4 → one real root, two complex
//     xa_xb =  vec2(-2.0, -0.0);
//     c = vec4(4.0, 4.0, 1.0, 1.0);
//     test[4] = is_cubic_solvable(c, 0.0, xa_xb); 

//     // Case 5: ill-conditioned roots: (x-1)(x-1.00001)(x-2)
//     xa_xb =  vec2(0.0, 3.0);
//     c = vec4(-2.00002 , 5.00003, -4.00001, 1.0);
//     test[5] = is_cubic_solvable(c, 0.0, xa_xb);

//     return test[0] && test[1] && test[2] && test[3] && test[4] && test[5];
// }

// bool test_cubic_3(out bool test[6])
// {
//     vec3 r; vec4 c;

//     // Case 0: (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6 → three real roots
//     r = vec3(1.0, 2.0, 3.0);
//     c = vec4(-6.0, 11.0, -6.0, 1.0);
//     test[0] = check_cubic_roots(c, 0.0, r);

//     // Case 1: (x-2)^3 = x^3 - 6x^2 + 12x - 8 → triple root
//     r = vec3(2.0, 2.0, 2.0);
//     c = vec4(-8.0, 12.0, -6.0, 1.0);
//     test[1] = check_cubic_roots(c, 0.0, r);

//     // Case 2: (x-1)^2(x+2) = x^3 - 3x^2 + 0x + 2 → double real root, single real root
//     r = vec3(-2.0, 1.0, 1.0);
//     c = vec4(2.0, 0.0, -3.0, 1.0);
//     test[2] = check_cubic_roots(c, 0.0, r);

//     // Case 3: x^3 = 0 → triple root at 0
//     r = vec3(0.0, 0.0, 0.0);
//     c = vec4(0.0, 0.0, 0.0, 1.0);
//     test[3] = check_cubic_roots(c, 1.0, r);

//     // Case 4: (x + 1)(x^2 + 4) = x^3 + x^2 + 4x + 4 → one real root, two complex
//     r = vec3(-1.0, 0.0, 0.0); // Placeholder roots
//     c = vec4(4.0, 4.0, 1.0, 1.0);
//     test[4] = check_cubic_roots(c, 0.0, r); 

//     // Case 5: ill-conditioned roots: (x-1)(x-1.000001)(x-2)
//     r = vec3(1.0, 1.000001, 2.0);
//     c = vec4(-2.000002, 5.000003, -4.000001, 1.0);
//     test[5] = check_cubic_roots(c, 0.0, r);

//     return test[0] && test[1] && test[2] && test[3] && test[4] && test[5];
// }

#endif