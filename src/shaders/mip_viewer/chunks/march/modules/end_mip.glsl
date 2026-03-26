
mip.terminated = mip.distance > ray.end_distance;
mip.position = distanceToPosition(mip.distance); 

mip.gradient = computeGradient(mip.position, mip.hessian);
mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);

