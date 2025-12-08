
// Compute quintic polynomial roots in [0, 1]
#if VARIATION_METHOD == 1
quintic_roots(quintic.roots, quintic.coeffs, 0.0, 1.0);

#elif VARIATION_METHOD == 2
quintic_roots_deflate(quintic.roots, quintic.coeffs, 0.0, 1.0);

#elif VARIATION_METHOD == 3
quintic_roots_deflate_inflate(quintic.roots, quintic.coeffs, 0.0, 1.0);

#elif VARIATION_METHOD == 4
quintic_roots_deflate_cubic(quintic.roots, quintic.coeffs, 0.0, 1.0);
#endif

quintic.root = mmin(quintic.roots);

// Compute derivative at root
eval_poly(quintic.coeffs, quintic.root, hit.derivative);
hit.derivative /= cell.span_distance;

// Compute orientation
hit.orientation = -sign(hit.derivative); 

// Compute intersection distance
hit.distance = mix(cell.entry_distance, cell.exit_distance, quintic.root);
hit.position = camera.position + ray.direction * hit.distance;

// Sample value
hit.value = sample_value_tricubic(hit.position);
hit.residue = hit.value - u_volume.isovalue;

// Compute gradients and hessian
hit.gradient = compute_gradient(hit.position, hit.hessian);

// Fix the orientation
hit.gradient *= hit.orientation; 
hit.hessian *= hit.orientation;

// Compute normal
hit.normal = normalize(hit.gradient);

// Compute principal curvatures
hit.curvatures = compute_curvatures(hit.gradient, hit.hessian);

