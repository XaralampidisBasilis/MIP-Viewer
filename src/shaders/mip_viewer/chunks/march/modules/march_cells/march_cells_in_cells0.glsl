
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

// update stats
#if DEBUG_ENABLED == 1

    stats.num_mips += 1;

#endif

// START_MARCH
const float eps = 0.001;
vec3 epsStep = u_ray.direction * eps;

for (int i = 0; i < MAX_CELLS; i++) 
{
    // UPDATE_CELL

    // compute next coordinates
    cell.coords = advanceCellCoords(cell.coords, cell.exit_step);

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

    // UPDATE_CUBIC     
    vec3 span_position = cell.exit_position - cell.entry_position;

    cubic.values.x = cubic.values.w;
    cubic.values.y = sampleVolume(cell.entry_position + span_position * (1.0 / 3.0));
    cubic.values.z = sampleVolume(cell.entry_position + span_position * (2.0 / 3.0));
    cubic.values.w = sampleVolume(cell.exit_position);

    #if DEBUG_ENABLED == 1

        stats.num_volume_fetches += 3;

    #endif

    // SOLVE_CUBIC
    cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
    CubicMax cubicMax = cubicMaxFromCoeffs_v2(cubic.coeffs);

    cubic.max_value = cubicMax.v;
    cubic.argmax_time = cubicMax.t;

    #if DEBUG_ENABLED == 1

        stats.num_cubics += 1;

    #endif

    // UPDATE_MIP
    if (mip.value < cubic.max_value) 
    {
        mip.distance = mix(cell.entry_distance, cell.exit_distance, cubicMax.t);
        mip.value = cubicMax.v;

        #if DEBUG_ENABLED == 1

            stats.num_mips += 1;

        #endif
    }

    if (cell.terminated) break;
}

// END_MIP
mip.position = rayDistanceToPosition(mip.distance); 
mip.gradient = compute_gradient(mip.position, mip.hessian);
mip.curvatures = compute_curvatures(mip.gradient, mip.hessian);
mip.normal = normalize(mip.gradient);


