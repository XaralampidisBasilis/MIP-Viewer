

// Compute position
mip.position = distanceToPosition(mip.distance); 

// Compute gradients and hessian
mip.gradient = computeGradient(mip.position, mip.hessian);

// Compute normal
mip.normal = normalize(mip.gradient);

// Compute principal curvatures
mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);

