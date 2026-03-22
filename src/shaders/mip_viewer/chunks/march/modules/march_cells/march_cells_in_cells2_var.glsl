
float eps_distance = u_ray.spacing * 0.001;
vec3 eps_direction = u_ray.direction * eps_distance;

// START_CELL
cell.coords = positionToCellCoords(ray.start_position + eps_direction);
cell.exit_distance = ray.start_distance;
cell.exit_position = ray.start_position; 
cell.exit_step = ivec3(0);
cell.empty = false;

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

for (int i = 0; i < MAX_CELLS; i++) 
{
    // UPDATE_CELL

    // Choose next current coords from either geometric exit or skip step
    cell.coords = advanceCellCoords(cell.coords, cell.exit_position + eps_direction, cell.step_radius, cell.exit_step);

    // Read skip radius and shadow flag for the current cell
    bool prev_empty = cell.empty;
    // cell.step_radius = sampleDistance5bit(cell.coords, cell.empty);
    cell.step_radius = sampleDistance8bit(cell.coords, cell.empty);

    // Current entry is the previous step's exit
    cell.entry_distance = cell.exit_distance;
    cell.entry_position = cell.exit_position;

    // Find exit point of the current skip cell
    cell.exit_distance = intersectCellExit(cell.coords, cell.step_radius, cell.exit_step);
    cell.exit_position = distanceToPosition(cell.exit_distance);

    // Distance covered inside this cell span
    cell.span_distance = cell.exit_distance - cell.entry_distance;

    // Stop once the ray exit goes beyond the ray end
    cell.terminated = cell.exit_distance > ray.end_distance - eps_distance;

    // Update traversal stats
    #if DEBUG_ENABLED == 1

        stats.num_distance_fetches += 1;
        stats.num_cells += 1;

    #endif

    if (!cell.empty) 
    {
        // SAMPLE CUBIC ALONG THE CURRENT NON-SHADOWED SPAN

        vec3 span_vector = cell.exit_position - cell.entry_position;

        // Reuse previous exit sample as the next entry sample when possible
        cubic.values.x = prev_empty ? sampleVolume(cell.entry_position) : cubic.values.w;
        cubic.values.y = sampleVolume(cell.entry_position + span_vector * (1.0 / 3.0));
        cubic.values.z = sampleVolume(cell.entry_position + span_vector * (2.0 / 3.0));
        cubic.values.w = sampleVolume(cell.exit_position);

        #if DEBUG_ENABLED == 1

            stats.num_volume_fetches += prev_empty ? 4 : 3;

        #endif

        // Convert sampled values to Bernstein coefficients for the max bound test
        cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;
        mip.update = any(greaterThan(cubic.bernstein_coeffs, vec4(mip.value)));

        if (mip.update)
        {
            // Compute exact cubic maximum only if the Bernstein bound can beat the MIP
            cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
            CubicMax cubic_max = cubicMaxOnUnitInterval(cubic.coeffs, cubic.values.x, cubic.values.w);

            cubic.max_value = cubic_max.v;
            cubic.argmax_time = cubic_max.t;

            #if DEBUG_ENABLED == 1

                stats.num_cubics += 1;

            #endif

            // Update MIP with the cubic maximum in this span
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