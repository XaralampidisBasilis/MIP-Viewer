
const float eps = 0.001;
vec3 epsStep = u_ray.direction * eps;

// START_BLOCK
block.coords = positionToBlockCoords(ray.start_position + epsStep);
block.exit_distance = ray.start_distance;
block.exit_position = ray.start_position; 
block.exit_step = ivec3(0);

// START_CUBIC
float ray_entry_value = sampleVolume(ray.start_position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;

#endif

// START_MIP
mip.distance = ray.start_distance;
mip.value = ray_entry_value;

#if DEBUG_ENABLED == 1

    stats.num_mips += 1;

#endif

// START_MARCH

for (int j = 0; j < MAX_CELLS; j++) 
{
    // UPDATE_BLOCK

    // Choose next block coords from either geometric exit or skip step
    block.coords = advanceBlockCoords(block.coords, block.exit_position + epsStep, block.step_radius, block.exit_step);

    // Read skip radius and shadow flag for the current block
    // block.step_radius = sampleDistance5bit(block.coords, block.shadowed);
    block.step_radius = sampleDistance8bit(block.coords, block.shadowed);

    // Current entry is the previous step's exit
    block.entry_distance = block.exit_distance;
    block.entry_position = block.exit_position;
    block.entry_step = block.exit_step;

    // Find exit point of the current skip block
    block.exit_distance = intersectBlockExit(block.coords, block.step_radius, block.exit_step);
    block.exit_position = rayDistanceToPosition(block.exit_distance);

    // Distance covered inside this block span
    block.span_distance = block.exit_distance - block.entry_distance;

    // Stop once the ray exit goes beyond the ray end
    block.terminated = block.exit_distance > ray.end_distance - eps;

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_blocks += 1;

    #endif

    if (block.shadowed && !block.terminated) 
    {
        continue;
    }

    float block_entry_value = sampleVolume(block.entry_position);

    // UPDATE_MIP
    if (mip.value < block_entry_value) 
    {
        mip.distance = block.entry_distance;
        mip.value = block_entry_value;

        #if DEBUG_ENABLED == 1

            stats.num_mips += 1;

        #endif
    }

    if (block.terminated) break;
}

// END_MIP
mip.position = rayDistanceToPosition(mip.distance); 
mip.gradient = computeGradient(mip.position, mip.hessian);
mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);


