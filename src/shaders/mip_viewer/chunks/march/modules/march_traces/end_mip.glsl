

// Compute position
mip.terminated = mip.distance > ray.end_distance - eps_distance;

if (mip.terminated)
{
    mip.distance = ray.end_distance;
    mip.value = sampleVolume(ray.end_position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;

    #endif
}
 
mip.position = distanceToPosition(mip.distance); 

// Compute gradients and hessian
mip.gradient = computeGradient(mip.position, mip.hessian);

// Compute normal
mip.normal = normalize(mip.gradient);

// Compute principal curvatures
mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);

