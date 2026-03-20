// START_CELL
cell.coords = positionToCellCoords(ray.start_position);
cell.exit_distance = ray.start_distance;
cell.exit_position = ray.start_position; 
cell.exit_step = ivec3(0);

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

// MARCH THROUGH CELLS ALONG THE RAY

const float eps = 0.001;
vec3 epsStep = u_ray.direction * eps;

bool prevNonShadowed = true;

for (int i = 0; i < MAX_CELLS; i++) 
{
    // UPDATE_CELL

    // Choose next current coords from either geometric exit or skip step
    cell.coords = advanceCellCoords(cell.coords, cell.exit_position + epsStep, cell.step_radius, cell.exit_step);

    // Read skip radius and shadow flag for the current cell
    cell.step_radius = sample_rgba16ui_distance_fast(cell.coords, cell.shadowed);
    // cell.step_radius = sample_rgb32ui_distance_fast(cell.coords, cell.shadowed);

    // Current entry is the previous step's exit
    cell.entry_distance = cell.exit_distance;
    cell.entry_position = cell.exit_position;

    // Find exit point of the current skip cell
    cell.exit_distance = intersectCellExit(cell.coords, cell.step_radius, cell.exit_step);
    cell.exit_position = rayDistanceToPosition(cell.exit_distance);

    // Distance covered inside this cell span
    cell.span_distance = cell.exit_distance - cell.entry_distance;

    // Stop once the ray exit goes beyond the ray end
    cell.terminated = cell.exit_distance > ray.end_distance - eps;

    // Update traversal stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_cells += 1;

    #endif

    bool consecutiveNonShadowed = prevNonShadowed && !cell.shadowed;
    prevNonShadowed = !cell.shadowed;

    if (!cell.shadowed) 
    {
        // SAMPLE CUBIC ALONG THE CURRENT NON-SHADOWED SPAN

        vec3 span_position = cell.exit_position - cell.entry_position;

        // Reuse previous exit sample as the next entry sample when possible
        cubic.values.x = consecutiveNonShadowed ? cubic.values.w : sampleVolume(cell.entry_position);
        cubic.values.y = sampleVolume(cell.entry_position + span_position * (1.0 / 3.0));
        cubic.values.z = sampleVolume(cell.entry_position + span_position * (2.0 / 3.0));
        cubic.values.w = sampleVolume(cell.exit_position);

        // Convert sampled values to Bernstein coefficients for the max bound test
        cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;

        #if DEBUG_ENABLED == 1

            stats.num_volume_fetches += consecutiveNonShadowed ? 3 : 4;

        #endif

        if (any(greaterThan(cubic.bernstein_coeffs, vec4(mip.value))))
        {
            // Compute exact cubic maximum only if the Bernstein bound can beat the MIP
            cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
            CubicMax cubicMax = cubicMaxFromCoeffs_v2(cubic.coeffs);

            cubic.max_value = cubicMax.v;
            cubic.argmax_time = cubicMax.t;

            #if DEBUG_ENABLED == 1

                stats.num_cubics += 1;

            #endif

            // Update MIP with the cubic maximum in this span
            mip.distance = mix(cell.entry_distance, cell.exit_distance, cubicMax.t);
            mip.value = cubicMax.v;

            #if DEBUG_ENABLED == 1

                stats.num_mips += 1;

            #endif
        }
    }

    if (cell.terminated) break;
}

// END_MIP
mip.position = rayDistanceToPosition(mip.distance); 
mip.gradient = compute_gradient(mip.position, mip.hessian);
mip.curvatures = compute_curvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);