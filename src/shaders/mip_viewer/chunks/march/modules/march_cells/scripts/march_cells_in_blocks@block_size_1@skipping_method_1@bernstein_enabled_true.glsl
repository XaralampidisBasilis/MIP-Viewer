

// START_BLOCK
block.coords = positionToBlockCoords(ray.start_position);
block.exit_distance = ray.start_distance;
block.exit_position = ray.start_position; 
block.exit_step = ivec3(0);
block.empty = false;

// START_CUBIC
cubic.values.w = sampleVolume(ray.start_position);

#if DEBUG_ENABLED == 1

    stats.num_volume_fetches += 1;

#endif

// START_MIP
mip.distance = ray.start_distance;
mip.value = cubic.values.w;

#if DEBUG_ENABLED == 1

    stats.num_mips += 1;

#endif

// START_MARCH

for (int j = 0; j < MAX_BLOCKS; j++) 
{
    // UPDATE_BLOCK

    // Choose next block coords from either geometric exit or skip step
    block.coords = advanceBlockCoords(block.coords, block.exit_step, block.step_radius, block.exit_position + ray.eps_direction);

    // Read skip radius and shadow flag for the current block
    block.prev_empty = block.empty;
    // block.step_radius = sampleDistance1bit(block.coords, block.empty);
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
    block.terminated = block.exit_distance > ray.end_distance;

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_blocks += 1;

    #endif

    if (block.empty)
    {
        if (!block.terminated) continue; else break;    
    }

    // START_CUBIC
    if(block.prev_empty)
    {
        cubic.values.w = sampleVolume(block.entry_position);

        #if DEBUG_ENABLED == 1

            stats.num_volume_fetches += 1;
        
        #endif
    }

    // UPDATE_CUBIC     
    vec3 span_vector = block.exit_position - block.entry_position;

    cubic.values.x = cubic.values.w;
    cubic.values.y = sampleVolume(block.entry_position + span_vector * (1.0 / 3.0));
    cubic.values.z = sampleVolume(block.entry_position + span_vector * (2.0 / 3.0));
    cubic.values.w = sampleVolume(block.exit_position);

    #if DEBUG_ENABLED == 1

        stats.num_volume_fetches += 3;

    #endif

    // MAXIMIZE_BERNSTEIN
    cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;
    mip.update = any(greaterThan(cubic.bernstein_coeffs, vec4(mip.value)));

    if (mip.update)
    {
        // MAXIMIZE_CUBIC
        cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
        CubicMax cubic_max = cubicMaxOnUnitInterval(cubic.coeffs, cubic.values.x, cubic.values.w);
        
        cubic.argmax_t = cubic_max.t;
        cubic.max_value = cubic_max.v;

        #if DEBUG_ENABLED == 1

            stats.num_cubics += 1;

        #endif

        mip.update = cubic.max_value > mip.value;
    }

    // UPDATE_MIP
    if (mip.update)
    {
        mip.distance = mix(block.entry_distance, block.exit_distance, cubic.argmax_t);
        mip.value = cubic.max_value;

        #if DEBUG_ENABLED == 1

            stats.num_mips += 1;

        #endif
    }

    if (block.terminated) break;
}

// END_MIP
mip.terminated = mip.distance > ray.end_distance;
mip.position = distanceToPosition(mip.distance); 
mip.gradient = computeGradient(mip.position, mip.hessian);
mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);


