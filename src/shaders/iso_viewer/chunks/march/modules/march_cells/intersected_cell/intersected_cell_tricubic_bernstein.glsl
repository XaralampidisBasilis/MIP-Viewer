
#include "../update_quintic"

// Convert the residual polynomial to Bernstein basis using precomputed transformation matrices
mat4x3 bernstein_coeffs = matrixCompMult(quad_bernstein * residuals * cubic_bernstein, quintic_bernstein_weights);

// Collapse the Bernstein coefficient matrix into a coefficients vector by summing anti-diagonals
sum_anti_diags(bernstein_coeffs, quintic.bernstein_coeffs);

// Compute sign change in berstein coefficients
if (sign_change(quintic.bernstein_coeffs))
{
    // Compute quintic coefficient matrix and sum the anti diagonals
    mat4x3 coeffs = quad_inv_vander * residuals * cubic_inv_vander;
    sum_anti_diags(coeffs, quintic.coeffs);

    // Compute quintic intersection by evaluating sign changes
    #if INTERSECTION_TEST == 0
    cell.intersected = sign_change(quintic.residuals) || quintic_has_root_sample(quintic.coeffs, 0.0, 1.0);

    #elif INTERSECTION_TEST == 1
    cell.intersected = sign_change(quintic.residuals) || quintic_has_root(quintic.coeffs, 0.0, 1.0);
    #endif

    #if DEBUG_ENABLED == 1
    stats.num_intersection_tests += 1;
    #endif
}

// update stats
#if DEBUG_ENABLED == 1
stats.num_texture_fetches += 3;
#endif


