
#include "../update_quintic"

// Reconstruct the quintic polynomial coefficients from the residual matrix
// Using inverse Vandermonde matrices for quadratic and cubic interpolation
mat4x3 coeffs = quad_inv_vander * residuals * cubic_inv_vander;

// Extract final quintic coefficients by summing the anti-diagonals of the matrix
// Each anti-diagonal corresponds to a coefficient basis term
sum_anti_diags(coeffs, quintic.coeffs);

// Perform root detection in [0,1] by checking sign changes:
// First on sampled residuals (fast), then refined on polynomial coefficients (fallback)
#if INTERSECTION_TEST == 0
cell.intersected = sign_change(quintic.residuals) || quintic_has_root_sample(quintic.coeffs, 0.0, 1.0);

#elif INTERSECTION_TEST == 1
cell.intersected = sign_change(quintic.residuals) || quintic_has_root(quintic.coeffs, 0.0, 1.0);
#endif

// Update fetch/test counters for performance statistics
#if DEBUG_ENABLED == 1
stats.num_texture_fetches += 3;
stats.num_intersection_tests += 1;
#endif
