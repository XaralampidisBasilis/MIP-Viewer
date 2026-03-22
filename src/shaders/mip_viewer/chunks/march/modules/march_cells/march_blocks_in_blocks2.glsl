
float eps_distance = u_ray.spacing * 0.001;
vec3 eps_direction = u_ray.direction * eps_distance;

// START_BLOCK
block.coords = positionToBlockCoords(ray.start_position + eps_direction);
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
    block.coords = advanceBlockCoords(block.coords, block.exit_step, block.step_radius, block.exit_position + eps_direction);

    // Read skip radius and shadow flag for the current block
    // block.step_radius = sampleDistance5bit(block.coords, block.empty);
    block.step_radius = sampleDistance8bit(block.coords, block.empty);

    // Current entry is the previous step's exit
    block.entry_distance = block.exit_distance;
    block.entry_position = block.exit_position;
    block.entry_step = block.exit_step;

    // Find exit point of the current skip block
    block.exit_distance = intersectBlockExit(block.coords, block.step_radius, block.exit_step);
    block.exit_position = distanceToPosition(block.exit_distance);

    // Distance covered inside this block span
    block.span_distance = block.exit_distance - block.entry_distance;

    // Stop once the ray exit goes beyond the ray end
    block.terminated = block.exit_distance > ray.end_distance - eps_distance;

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_blocks += 1;

    #endif

    if (block.empty && !block.terminated) 
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
mip.gradient = computeGradient(mip.position, mip.hessian);
mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);


