
#include "../update_cubic"

// from the sampled intensities we can compute the trilinear interpolation cubic polynomial coefficients
cubic.coeffs = cubic.residuals * cubic_inv_vander;

// check cubic intersection and sign crossings for degenerate cases
#if INTERSECTION_TEST == 0
cell.intersected = sign_change(cubic.residuals) || cubic_has_root(cubic.coeffs, 0.0, 1.0);

#elif INTERSECTION_TEST == 1
cell.intersected = sign_change(cubic.residuals) || cubic_has_root_sample(cubic.coeffs, 0.0, 1.0);
#endif

// update stats
#if DEBUG_ENABLED == 1
stats.num_texture_fetches += 3;
stats.num_intersection_tests += 1;
#endif
