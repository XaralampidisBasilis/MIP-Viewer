#ifndef TEST_QUARTIC
#define TEST_QUARTIC

#ifndef QUARTIC_ROOTS
#include "./quartic_roots"
#endif
#ifndef IS_QUARTIC_SOLVABLE
#include "./is_quadratic_solvable"
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

// Returns true if any roots satisfy f(x) ≈ 0

bvec4 test_quartic_residue(in float c[5], in float x0)
{
    // Get roots from your solver
    vec4 x0_x1_x2_x3 = quartic_roots(c, x0);
    sort(x0_x1_x2_x3);

    // Evaluate the original polynomial at each root
    vec4 y0_y1_y2_y3 = eval_poly(c, x0_x1_x2_x3);

    // Check if values are near zero
    bvec4 b0_b1_b2_b3 = lessThan(abs(y0_y1_y2_y3), vec4(MILLI_TOLERANCE));

    // Check if any value is near zero
    return b0_b1_b2_b3;
}

// Checks if computed roots ≈ expected ones

bvec4 test_quartic_roots(in float c[5], in float x0, in vec4 r0_r1_r2_r3)
{
    // Get roots from your solver
    vec4 x0_x1_x2_x3 = quartic_roots(c, x0);

    // Sort roots in ascending form
    sort(x0_x1_x2_x3);
    sort(r0_r1_r2_r3);

    // Check if roots are approximated well
    bvec4 b0_b1_b2_b3 = lessThan(abs(x0_x1_x2_x3 - r0_r1_r2_r3), vec4(MILLI_TOLERANCE));

    // Return true if all roots are approximated
    return b0_b1_b2_b3;
}

// Checks quartic roots for different cases

bool test_quartic4_residue(out bvec4 test[8])
{
    float c[5];

    // Case 0: (x - 1)*(x - 2)*(x - 3)*(x - 4)
    // = x^4 - 10x^3 + 35x^2 - 50x + 24
    c = float[5](24.0, -50.0, 35.0, -10.0, 1.0);
    test[0] = test_quartic_residue(c, 0.0);

    // Case 1: (x - 2)*(x - 4)*(x - 6)*(x - 8)
    // = x^4 - 20x^3 + 140^2 - 440x + 384
    c = float[5](384.0, -400.0, 140.0, -20.0, 1.0);
    test[1] = test_quartic_residue(c, 0.0);

    // Case 2: (x + 1)*(x + 2)*(x + 3)*(x + 4)
    // = x^4 + 10x^3 + 35x^2 + 50x + 24
    c = float[5](24.0, 50.0, 35.0, 10.0, 1.0);
    test[2] = test_quartic_residue(c, 0.0);

    // Case 3: (x - 0.5)*(x - 1.5)*(x - 2.5)*(x - 3.5)
    // = x^4 - 8x^3 + 21.5x^2 - 22x + 6.5625
    c = float[5](6.5625, -22.0, 21.5, -8.0, 1.0);
    test[3] = test_quartic_residue(c, 0.0);

    // Case 4: (x - 10)*(x - 11)*(x - 12)*(x - 13)
    // = x^4 - 46x^3 + 791x^2 - 6026 + 17160
    c = float[5](17160.0, -6026.0, 791.0, -46.0, 1.0);
    test[4] = test_quartic_residue(c, 0.0);

    // Case 5: (x + 0.1)*(x + 0.2)*(x + 0.3)*(x + 0.4)
    // = x^4 + 1.0x^3 + 0.35x^2 + 0.05x + 0.0024
    c = float[5](0.0024, 0.05, 0.35, 1.0, 1.0);
    test[5] = test_quartic_residue(c, -1.0);

    // Case 6: (x - 100)*(x - 200)*(x - 300)*(x - 400)
    // = x^4 - 1000x^3 + 350000^2 - 50000000x + 2400000000
    c = float[5](2400000000.0, -50000000.0, 350000.0, -1000.0, 1.0);
    test[6] = test_quartic_residue(c, 0.0);

    // Case 7: (x - 1)*(x - 1.00001)*(x - 2)*(x - 2.00001)
    c = float[5](4.0000600002, -12.0001300003, 13.0000900001, -6.00002, 1.0);
    test[7] = test_quartic_residue(c, 0.0);

    return all(test[0]) && all(test[1]) && all(test[2]) && all(test[3]) && all(test[4]) && all(test[5]) && all(test[6]) && all(test[7]);
}

bool test_quartic4_roots(out bvec4 test[8])
{
    vec4 r; float c[5];
    
    // Case 0: (x - 1)*(x - 2)*(x - 3)*(x - 4)
    r = vec4(1.0, 2.0, 3.0, 4.0);
    c = float[5](24.0, -50.0, 35.0, -10.0, 1.0);
    test[0] = test_quartic_roots(c, 0.0, r);

    // Case 1: (x - 2)*(x - 4)*(x - 6)*(x - 8)
    r = vec4(2.0, 4.0, 6.0, 8.0);
    c = float[5](384.0, -400.0, 140.0, -20.0, 1.0);
    test[1] = test_quartic_roots(c, 0.0, r);

    // Case 2: (x + 1)*(x + 2)*(x + 3)*(x + 4)
    r = vec4(-4.0, -3.0, -2.0, -1.0);
    c = float[5](24.0, 50.0, 35.0, 10.0, 1.0);
    test[2] = test_quartic_roots(c, 0.0, r);

    // Case 3: (x - 0.5)*(x - 1.5)*(x - 2.5)*(x - 3.5)
    r = vec4(0.5, 1.5, 2.5, 3.5);
    c = float[5](6.5625, -22.0, 21.5, -8.0, 1.0);
    test[3] = test_quartic_roots(c, 0.0, r);

    // Case 4: (x - 10)*(x - 11)*(x - 12)*(x - 13)
    r = vec4(10.0, 11.0, 12.0, 13.0);
    c = float[5](17160.0, -6026.0, 791.0, -46.0, 1.0);
    test[4] = test_quartic_roots(c, 0.0, r);

    // Case 5: (x + 0.1)*(x + 0.2)*(x + 0.3)*(x + 0.4)
    r = vec4(-0.4, -0.3, -0.2, -0.1);
    c = float[5](0.0024, 0.05, 0.35, 1.0, 1.0);
    test[5] = test_quartic_roots(c, -1.0, r);

    // Case 6: (x - 100)*(x - 200)*(x - 300)*(x - 400)
    r = vec4(100.0, 200.0, 300.0, 400.0);
    c = float[5](2400000000.0, -50000000.0, 350000.0, -1000.0, 1.0);
    test[6] = test_quartic_roots(c, 0.0, r);

    // Case 7: (x - 1)*(x - 1.00001)*(x - 2)*(x - 2.00001)
    r = vec4(1.0, 1.00001, 2.0, 2.00001);
    c = float[5](4.0000600002, -12.0001300003, 13.0000900001, -6.00002, 1.0);
    test[7] = test_quartic_roots(c, 0.0, r);

    return all(test[0]) && all(test[1]) && all(test[2]) && all(test[3]) && all(test[4]) && all(test[5]) && all(test[6]) && all(test[7]);
}

// bool test_quartic_1(out bool test[8])
// {
//     // Test results
//     vec4 r; float c[5];

//     // Case 0: (x-1)(x-2)(x-3)(x-4) = x^4 - 10x^3 + 35x^2 - 50x + 24 → four real roots
//     r = vec4(1.0, 2.0, 3.0, 4.0);
//     c = float[5](24.0, -50.0, 35.0, -10.0, 1.0);
//     test[0] = test_quartic_roots(c, 0.0, r);


//     // Case 1: (x-1)^3(x-2) = x^4 - 5x^3 + 9x^2 - 7x + 2 → one triple root and a single
//     r = vec4(1.0, 1.0, 1.0, 2.0);
//     c = float[5](2.0, -7.0, 9.0, -5.0, 1.0);
//     test[1] = test_quartic_roots(c, 0.0, r);


//     // Case 2: (x-1)^2(x-2)^2 = x^4 - 6x^3 + 13x^2 - 12x + 4 → two double roots
//     r = vec4(1.0, 1.0, 2.0, 2.0);
//     c = float[5](4.0, -12.0, 13.0, -6.0, 1.0);
//     test[2] = test_quartic_roots(c, 0.0, r);


//     // Case 3: x^4 = 0 → 4 repeated roots at 0
//     c = float[5](0.0, 0.0, 0.0, 0.0, 1.0);
//     r = vec4(0.0, 0.0, 0.0, 0.0);
//     test[3] = test_quartic_roots(c, 1.0, r);


//     // Case 4: (x² + 1)(x² + 4) = x⁴ + 5x² + 4 → two complex conjugate pairs
//     r = vec4(0.0); 
//     c = float[5](4.0, 0.0, 5.0, 0.0, 1.0);
//     test[4] = test_quartic_roots(c, 0.0, r);


//     // Case 5: (x - 2)²(x² + 1) = x⁴ - 4x³ + 5x² - 4x + 4 → one double real root, two complex
//     r = vec4(0.0, 0.0, 2.0, 2.0); 
//     c = float[5](4.0, -4.0, 5.0, -4.0, 1.0);
//     test[5] = test_quartic_roots(c, 0.0, r);


//     // Case 6: x(x - 1)(x - 2)(x - 3) = x⁴ - 6x³ + 11x² - 6x → quartic with a zero root
//     r = vec4(0.0, 1.0, 2.0, 3.0);
//     c = float[5](0.0, -6.0, 11.0, -6.0, 1.0);
//     test[6] = test_quartic_roots(c, 0.0, r);


//     // Case 7: (x - 1)(x - 1.00001)(x - 2)(x - 2.00001) → numerically ill-conditioned roots
//     r = vec4(1.0, 1.00001, 2.0, 2.00001);
//     c = float[5](4.00006, -12.0001, 13.0001, -6.00002, 1.0);
//     test[7] = test_quartic_roots(c, 0.0, r);

//     // Combine all tests
//     return test[0] && test[1] && test[2] && test[3] && test[4] && test[5] && test[6] && test[7];
// }

#endif