const float eps = 0.001;
vec3 epsStep = u_ray.direction * eps;

// START_CELL
cell.coords = positionToCellCoords(ray.start_position + epsStep);
cell.exit_distance = ray.start_distance;
cell.exit_position = ray.start_position; 
cell.exit_step = ivec3(0);
cell.shadowed = false;

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

bool prevNonShadowed = true;

for (int i = 0; i < MAX_CELLS; i++) 
{
    // UPDATE_CELL

    // compute next coordinates
    cell.coords = advanceCellCoords(cell.coords, cell.exit_step);

    // compute shadowed
    bool prev_shadowed = cell.shadowed;
    cell.shadowed = sampleShadow(cell.coords);

    // compute entry from previous exit
    cell.entry_distance = cell.exit_distance;
    cell.entry_position = cell.exit_position;

    // compute exit from cell ray intersection 
    cell.exit_distance = intersectCellExit(cell.coords, cell.exit_step);
    cell.exit_position = rayDistanceToPosition(cell.exit_distance);

    // compute span distance
    cell.span_distance = cell.exit_distance - cell.entry_distance;

    // compute termination condition
    cell.terminated = cell.exit_distance > ray.end_distance - eps;

    // update stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_cells += 1;

    #endif

    if (!cell.shadowed) 
    {
        // UPDATE_CUBIC     
        vec3 span_vector = cell.exit_position - cell.entry_position;

        cubic.values.x = prev_shadowed ? sampleVolume(cell.entry_position) : cubic.values.w;
        cubic.values.y = sampleVolume(cell.entry_position + span_vector * (1.0 / 3.0));
        cubic.values.z = sampleVolume(cell.entry_position + span_vector * (2.0 / 3.0));
        cubic.values.w = sampleVolume(cell.exit_position);

        #if DEBUG_ENABLED == 1

            stats.num_volume_fetches += prev_shadowed ? 4 : 3;

        #endif

        // BERNSTEIN_TEST
        cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;
        bool update_mip = any(greaterThan(cubic.bernstein_coeffs, vec4(mip.value)));

        if (update_mip)
        {
            // SOLVE CUBIC
            cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
            CubicMax cubic_max = cubicMaxFromCoeffs(cubic.coeffs);

            cubic.max_value = cubic_max.v;
            cubic.argmax_time = cubic_max.t;
            
            #if DEBUG_ENABLED == 1

                stats.num_cubics += 1;

            #endif

            // UPDATE_MIP
            mip.distance = mix(cell.entry_distance, cell.exit_distance, cubic_max.t);
            mip.value = cubic_max.v;

            #if DEBUG_ENABLED == 1

                stats.num_mips += 1;

            #endif
        }
    }

    if (cell.terminated) break;
}

// END_MIP
mip.position = rayDistanceToPosition(mip.distance); 
mip.gradient = computeGradient(mip.position, mip.hessian);
mip.curvatures = computePrincipalCurvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);


