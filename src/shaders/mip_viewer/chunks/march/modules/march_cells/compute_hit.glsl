
// Compute cubic polynomial roots in [0, 1]
#if VARIATION_METHOD == 1
cubic_roots(cubic.roots, cubic.coeffs, 0.0, 1.0);

#elif VARIATION_METHOD == 2
cubic_roots_deflate(cubic.roots, cubic.coeffs, 0.0, 1.0);
#endif

cubic.root = mmin(cubic.roots);

// Compute derivative at root
eval_poly(cubic.coeffs, cubic.root, hit.derivative);
hit.derivative /= cell.span_distance;

// Compute orientation
hit.orientation = -sign(hit.derivative); 

// Compute intersection distance/position
hit.distance = mix(cell.entry_distance, cell.exit_distance, cubic.root);
hit.position = camera.position + ray.direction * hit.distance;

// Sample value/residue
hit.value = sample_value_trilinear(hit.position);
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


