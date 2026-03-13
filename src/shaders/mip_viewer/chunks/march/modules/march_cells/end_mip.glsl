
// Compute position
mip.position = ray.origin + ray.direction * mip.distance; 

// Compute gradients and hessian
mip.gradient = compute_gradient(mip.position, mip.hessian);

// Compute normal
mip.normal = normalize(mip.gradient);

// Compute principal curvatures
mip.curvatures = compute_curvatures(mip.gradient, mip.hessian);

