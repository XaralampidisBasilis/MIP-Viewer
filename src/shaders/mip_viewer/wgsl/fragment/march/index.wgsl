fn snap_trace_distance_ceil(t: f32, step_distance: f32, phase: f32) -> f32
{
    return step_distance * (ceil(t / step_distance - phase) + phase);
}

fn position_to_cell_coords(position: vec3<f32>) -> vec3<i32>
{
    return vec3<i32>(floor(position + vec3<f32>(0.5)));
}

fn position_to_block_coords(position: vec3<f32>, block_size: i32) -> vec3<i32>
{
    return position_to_cell_coords(position) / vec3<i32>(max(block_size, 1));
}

fn cell_coords_to_min_position(coords: vec3<i32>) -> vec3<f32>
{
    return vec3<f32>(coords) - vec3<f32>(0.5);
}

fn block_coords_to_min_position(coords: vec3<i32>, block_size: i32) -> vec3<f32>
{
    return vec3<f32>(coords * vec3<i32>(max(block_size, 1))) - vec3<f32>(0.5);
}

fn cell_coords_to_far_distances(coords: vec3<i32>, ray: RayState) -> vec3<f32>
{
    let c_min = cell_coords_to_min_position(coords);
    let c_max = c_min + vec3<f32>(1.0);
    let c_far = vec3<f32>(
        select(c_min.x, c_max.x, ray.sign_direction.x > 0),
        select(c_min.y, c_max.y, ray.sign_direction.y > 0),
        select(c_min.z, c_max.z, ray.sign_direction.z > 0)
    );
    return (c_far - ray.origin) * ray.inv_direction;
}

fn advance_cell_coords(coords: vec3<i32>, exit_step: vec3<i32>, ray: RayState) -> vec3<i32>
{
    return coords + exit_step * ray.sign_direction;
}

fn advance_cell_far_distances(t_far: vec3<f32>, exit_step: vec3<i32>, ray: RayState) -> vec3<f32>
{
    return t_far + vec3<f32>(exit_step) * ray.step_distances;
}

fn intersect_cell_far_distances(t_far: vec3<f32>) -> ExitHit
{
    let t_exit = min(min(t_far.x, t_far.y), t_far.z);
    return ExitHit(
        t_exit,
        vec3<i32>(
            select(0, 1, t_far.x == t_exit),
            select(0, 1, t_far.y == t_exit),
            select(0, 1, t_far.z == t_exit)
        )
    );
}

fn start_cell_in_ray(ray: RayState) -> CellState
{
    let coords = position_to_cell_coords(ray.start_position);
    return CellState(
        false,
        coords,
        cell_coords_to_far_distances(coords, ray),
        ray.start_distance,
        ray.start_distance,
        ray.start_position,
        ray.start_position,
        vec3<i32>(0)
    );
}

fn start_cell_coords_in_block(block_coords: vec3<i32>, entry_step: vec3<i32>, entry_position: vec3<f32>, block_size: i32) -> vec3<i32>
{
    let entry_coords = position_to_cell_coords(entry_position);
    let step_coords = block_coords * vec3<i32>(max(block_size, 1));
    return entry_coords + (step_coords - entry_coords) * entry_step;
}

fn start_cell_in_block(block: BlockState, ray: RayState, block_size: i32) -> CellState
{
    let coords = start_cell_coords_in_block(block.coords, block.entry_step, block.entry_position + ray.eps_direction, block_size);
    return CellState(
        false,
        coords,
        cell_coords_to_far_distances(coords, ray),
        block.entry_distance,
        block.entry_distance,
        block.entry_position,
        block.entry_position,
        vec3<i32>(0)
    );
}

fn update_cell_in_ray(cell_in: CellState, ray: RayState) -> CellState
{
    var cell = cell_in;
    cell.coords = advance_cell_coords(cell.coords, cell.exit_step, ray);
    cell.far_distances = advance_cell_far_distances(cell.far_distances, cell.exit_step, ray);
    cell.entry_distance = cell.exit_distance;
    cell.entry_position = cell.exit_position;

    let exit = intersect_cell_far_distances(cell.far_distances);
    cell.exit_distance = exit.distance;
    cell.exit_step = exit.step;
    cell.exit_position = distance_to_position(ray, cell.exit_distance);
    cell.terminated = cell.exit_distance > ray.end_distance - ray.eps_distance;

    if (cell.terminated) {
        cell.exit_distance = ray.end_distance;
        cell.exit_position = ray.end_position;
    }

    return cell;
}

fn update_cell_in_block(cell_in: CellState, block: BlockState, ray: RayState, block_size: i32) -> CellState
{
    var cell = cell_in;

    if (block_size != 1) {
        cell.coords = advance_cell_coords(cell.coords, cell.exit_step, ray);
        cell.far_distances = advance_cell_far_distances(cell.far_distances, cell.exit_step, ray);
        cell.entry_distance = cell.exit_distance;
        cell.entry_position = cell.exit_position;

        let exit = intersect_cell_far_distances(cell.far_distances);
        cell.exit_distance = exit.distance;
        cell.exit_step = exit.step;
        cell.exit_position = distance_to_position(ray, cell.exit_distance);
        cell.terminated = cell.exit_distance > block.exit_distance - ray.eps_distance;

        if (cell.terminated) {
            cell.exit_distance = block.exit_distance;
            cell.exit_position = block.exit_position;
        }
    } else {
        cell.entry_distance = block.entry_distance;
        cell.entry_position = block.entry_position;
        cell.exit_distance = block.exit_distance;
        cell.exit_position = block.exit_position;
        cell.terminated = true;
    }

    return cell;
}

fn advance_block_coords(coords: vec3<i32>, exit_step: vec3<i32>, ray: RayState) -> vec3<i32>
{
    return coords + exit_step * ray.sign_direction;
}

fn advance_block_coords_distance(
    coords: vec3<i32>,
    exit_step: vec3<i32>,
    step_radius: i32,
    exit_position: vec3<f32>,
    block_size: i32,
    ray: RayState
) -> vec3<i32>
{
    if (step_radius == 1) {
        return advance_block_coords(coords, exit_step, ray);
    }

    let exit_coords = position_to_block_coords(exit_position, block_size);
    let step_coords = coords + vec3<i32>(step_radius) * ray.sign_direction;
    return exit_coords + (step_coords - exit_coords) * exit_step;
}

fn intersect_block_exit(coords: vec3<i32>, radius: i32, block_size: i32, ray: RayState) -> ExitHit
{
    let c_min = coords - vec3<i32>(radius) + vec3<i32>(1);
    let c_max = coords + vec3<i32>(radius);
    let b_min = block_coords_to_min_position(c_min, block_size);
    let b_max = block_coords_to_min_position(c_max, block_size);
    let b_exit = vec3<f32>(
        select(b_min.x, b_max.x, ray.sign_direction.x > 0),
        select(b_min.y, b_max.y, ray.sign_direction.y > 0),
        select(b_min.z, b_max.z, ray.sign_direction.z > 0)
    );
    let t_far = (b_exit - ray.origin) * ray.inv_direction;
    let t_exit = min(min(t_far.x, t_far.y), t_far.z);
    return ExitHit(
        t_exit,
        vec3<i32>(
            select(0, 1, t_far.x == t_exit),
            select(0, 1, t_far.y == t_exit),
            select(0, 1, t_far.z == t_exit)
        )
    );
}

fn start_block_in_ray(ray: RayState, block_size: i32) -> BlockState
{
    return BlockState(
        false,
        false,
        false,
        position_to_block_coords(ray.start_position, block_size),
        1,
        ray.start_distance,
        ray.start_distance,
        ray.start_position,
        ray.start_position,
        vec3<i32>(0),
        vec3<i32>(0)
    );
}

fn update_block_in_ray(
    block_in: BlockState,
    distance_words: ptr<storage, array<u32>, read>,
    distance_dimensions: vec3<i32>,
    dominant_axis: u32,
    quadrant_index: u32,
    group_index: u32,
    distance_variation: i32,
    skipping_method: i32,
    block_size: i32,
    words_per_voxel: u32,
    ray: RayState
) -> BlockState
{
    var block = block_in;

    if (skipping_method == 1) {
        block.coords = advance_block_coords_distance(block.coords, block.exit_step, block.step_radius, block.exit_position + ray.eps_direction, block_size, ray);
    } else {
        block.coords = advance_block_coords(block.coords, block.exit_step, ray);
    }

    block.prev_empty = block.empty;

    let distance_sample = sample_distance(
        distance_words,
        block.coords,
        distance_dimensions,
        dominant_axis,
        quadrant_index,
        group_index,
        distance_variation,
        words_per_voxel
    );
    block.empty = distance_sample.empty;
    block.step_radius = select(1, distance_sample.distance, skipping_method == 1);

    block.entry_distance = block.exit_distance;
    block.entry_position = block.exit_position;
    block.entry_step = block.exit_step;

    let exit = intersect_block_exit(block.coords, block.step_radius, block_size, ray);
    block.exit_distance = exit.distance;
    block.exit_step = exit.step;
    block.exit_position = distance_to_position(ray, block.exit_distance);
    block.terminated = block.exit_distance > ray.end_distance - ray.eps_distance;

    if (block.terminated) {
        block.exit_distance = ray.end_distance;
        block.exit_position = ray.end_position;
    }

    return block;
}

fn update_mip(mip_in: MipState, value: f32, distance: f32, position: vec3<f32>) -> MipState
{
    var mip = mip_in;

    if (value > mip.value) {
        mip.value = value;
        mip.distance = distance;
        mip.position = position;
    }

    return mip;
}

fn make_initial_trace_mip(
    volume_map: texture_3d<f32>,
    volume_sampler: sampler,
    volume_inv_dimensions: vec3<f32>,
    ray: RayState,
    max_traces_in_cell: i32
) -> MipState
{
    let trace_step = ray.step_distance / f32(max(max_traces_in_cell - 1, 1));
    let trace_distance = snap_trace_distance_ceil(ray.start_distance, trace_step, ray.phase);
    let trace_position = distance_to_position(ray, trace_distance);
    let trace_value = sample_volume(volume_map, volume_sampler, trace_position, volume_inv_dimensions);
    return MipState(trace_value, trace_distance, trace_position);
}

fn march_traces_in_cells(
    volume_map: texture_3d<f32>,
    volume_sampler: sampler,
    volume_inv_dimensions: vec3<f32>,
    ray: RayState,
    max_cells: i32,
    max_traces_in_cell: i32
) -> MipState
{
    var cell = start_cell_in_ray(ray);
    var mip = make_initial_trace_mip(volume_map, volume_sampler, volume_inv_dimensions, ray, max_traces_in_cell);

    for (var j = 0; j < max(max_cells, 1); j = j + 1) {
        cell = update_cell_in_ray(cell, ray);

        for (var i = 1; i < max(max_traces_in_cell, 1); i = i + 1) {
            let alpha = f32(i) / f32(max(max_traces_in_cell - 1, 1));
            let trace_distance = mix(cell.entry_distance, cell.exit_distance, alpha);
            let trace_position = distance_to_position(ray, trace_distance);
            let trace_value = sample_volume(volume_map, volume_sampler, trace_position, volume_inv_dimensions);
            mip = update_mip(mip, trace_value, trace_distance, trace_position);
        }

        if (cell.terminated) {
            break;
        }
    }

    mip.position = distance_to_position(ray, mip.distance);
    return mip;
}

fn march_traces_in_cells_in_blocks(
    volume_map: texture_3d<f32>,
    volume_sampler: sampler,
    distance_words: ptr<storage, array<u32>, read>,
    volume_inv_dimensions: vec3<f32>,
    distance_dimensions: vec3<i32>,
    ray: RayState,
    dominant_axis: u32,
    quadrant_index: u32,
    group_index: u32,
    distance_variation: i32,
    skipping_method: i32,
    block_size: i32,
    max_blocks: i32,
    max_cells_in_block: i32,
    max_traces_in_cell: i32,
    words_per_voxel: u32
) -> MipState
{
    var block = start_block_in_ray(ray, block_size);
    var mip = make_initial_trace_mip(volume_map, volume_sampler, volume_inv_dimensions, ray, max_traces_in_cell);

    for (var k = 0; k < max(max_blocks, 1); k = k + 1) {
        block = update_block_in_ray(
            block,
            distance_words,
            distance_dimensions,
            dominant_axis,
            quadrant_index,
            group_index,
            distance_variation,
            skipping_method,
            block_size,
            words_per_voxel,
            ray
        );

        if (block.empty) {
            if (!block.terminated) {
                continue;
            }
            break;
        }

        var cell = start_cell_in_block(block, ray, block_size);

        for (var j = 0; j < max(max_cells_in_block, 1); j = j + 1) {
            cell = update_cell_in_block(cell, block, ray, block_size);

            for (var i = 1; i < max(max_traces_in_cell, 1); i = i + 1) {
                let alpha = f32(i) / f32(max(max_traces_in_cell - 1, 1));
                let trace_distance = mix(cell.entry_distance, cell.exit_distance, alpha);
                let trace_position = distance_to_position(ray, trace_distance);
                let trace_value = sample_volume(volume_map, volume_sampler, trace_position, volume_inv_dimensions);
                mip = update_mip(mip, trace_value, trace_distance, trace_position);
            }

            if (cell.terminated) {
                break;
            }
        }

        if (block.terminated) {
            break;
        }
    }

    mip.position = distance_to_position(ray, mip.distance);
    return mip;
}

fn cubic_coeffs(values: vec4<f32>) -> vec4<f32>
{
    return vec4<f32>(
        values.x,
        (-11.0 * values.x + 18.0 * values.y - 9.0 * values.z + 2.0 * values.w) * 0.5,
        (18.0 * values.x - 45.0 * values.y + 36.0 * values.z - 9.0 * values.w) * 0.5,
        (-9.0 * values.x + 27.0 * values.y - 27.0 * values.z + 9.0 * values.w) * 0.5
    );
}

fn cubic_bernstein_coeffs(values: vec4<f32>) -> vec4<f32>
{
    return vec4<f32>(
        values.x,
        (-5.0 * values.x + 18.0 * values.y - 9.0 * values.z + 2.0 * values.w) / 6.0,
        (2.0 * values.x - 9.0 * values.y + 18.0 * values.z - 5.0 * values.w) / 6.0,
        values.w
    );
}

fn eval_cubic(c: vec4<f32>, t: f32) -> f32
{
    return ((c.w * t + c.z) * t + c.y) * t + c.x;
}

fn sign_non_zero(x: f32) -> f32
{
    return select(-1.0, 1.0, x >= 0.0);
}

fn cubic_max_on_unit_interval(c: vec4<f32>, v0: f32, v1: f32) -> CubicMax
{
    var best_v = v0;
    var best_t = 0.0;

    if (v1 > best_v) {
        best_v = v1;
        best_t = 1.0;
    }

    let d = c.y;
    let b = 2.0 * c.z;
    let a = 3.0 * c.w;
    let disc = b * b - 4.0 * a * d;

    if (disc < 0.0) {
        return CubicMax(best_v, best_t);
    }

    let s = sqrt(disc);
    let q = -0.5 * (b + sign_non_zero(b) * s);

    let t0 = d / q;
    let t1 = q / a;
    let dd0 = 2.0 * a * t0 + b;
    let t = select(t1, t0, dd0 < 0.0);

    if (t > 0.0 && t < 1.0) {
        let v = eval_cubic(c, t);
        if (v > best_v) {
            best_v = v;
            best_t = t;
        }
    }

    return CubicMax(best_v, best_t);
}

fn update_cubic_mip(
    volume_map: texture_3d<f32>,
    volume_sampler: sampler,
    volume_inv_dimensions: vec3<f32>,
    ray: RayState,
    cell: CellState,
    cubic_w: f32,
    mip_in: MipState
) -> vec4<f32>
{
    let span_vector = cell.exit_position - cell.entry_position;
    let values = vec4<f32>(
        cubic_w,
        sample_volume(volume_map, volume_sampler, cell.entry_position + span_vector * (1.0 / 3.0), volume_inv_dimensions),
        sample_volume(volume_map, volume_sampler, cell.entry_position + span_vector * (2.0 / 3.0), volume_inv_dimensions),
        sample_volume(volume_map, volume_sampler, cell.exit_position, volume_inv_dimensions)
    );

    var mip = mip_in;
    let bernstein = cubic_bernstein_coeffs(values);

    if (any(bernstein > vec4<f32>(mip.value))) {
        let max_sample = cubic_max_on_unit_interval(cubic_coeffs(values), values.x, values.w);
        if (max_sample.value > mip.value) {
            mip.value = max_sample.value;
            mip.distance = mix(cell.entry_distance, cell.exit_distance, max_sample.point);
            mip.position = distance_to_position(ray, mip.distance);
        }
    }

    return vec4<f32>(values.w, mip.value, mip.distance, 0.0);
}

fn march_cells_in_cells(
    volume_map: texture_3d<f32>,
    volume_sampler: sampler,
    volume_inv_dimensions: vec3<f32>,
    ray: RayState,
    max_cells: i32
) -> MipState
{
    var cell = start_cell_in_ray(ray);
    var cubic_w = sample_volume(volume_map, volume_sampler, ray.start_position, volume_inv_dimensions);
    var mip = MipState(cubic_w, ray.start_distance, ray.start_position);

    for (var i = 0; i < max(max_cells, 1); i = i + 1) {
        cell = update_cell_in_ray(cell, ray);
        let packed = update_cubic_mip(volume_map, volume_sampler, volume_inv_dimensions, ray, cell, cubic_w, mip);
        cubic_w = packed.x;
        mip.value = packed.y;
        mip.distance = packed.z;
        mip.position = distance_to_position(ray, mip.distance);

        if (cell.terminated) {
            break;
        }
    }

    mip.position = distance_to_position(ray, mip.distance);
    return mip;
}

fn march_cells_in_blocks(
    volume_map: texture_3d<f32>,
    volume_sampler: sampler,
    distance_words: ptr<storage, array<u32>, read>,
    volume_inv_dimensions: vec3<f32>,
    distance_dimensions: vec3<i32>,
    ray: RayState,
    dominant_axis: u32,
    quadrant_index: u32,
    group_index: u32,
    distance_variation: i32,
    skipping_method: i32,
    block_size: i32,
    max_blocks: i32,
    max_cells_in_block: i32,
    words_per_voxel: u32
) -> MipState
{
    var block = start_block_in_ray(ray, block_size);
    var cubic_w = sample_volume(volume_map, volume_sampler, ray.start_position, volume_inv_dimensions);
    var mip = MipState(cubic_w, ray.start_distance, ray.start_position);

    for (var j = 0; j < max(max_blocks, 1); j = j + 1) {
        block = update_block_in_ray(
            block,
            distance_words,
            distance_dimensions,
            dominant_axis,
            quadrant_index,
            group_index,
            distance_variation,
            skipping_method,
            block_size,
            words_per_voxel,
            ray
        );

        if (block.empty) {
            if (!block.terminated) {
                continue;
            }
            break;
        }

        if (block.prev_empty) {
            cubic_w = sample_volume(volume_map, volume_sampler, block.entry_position, volume_inv_dimensions);
        }

        var cell = start_cell_in_block(block, ray, block_size);

        for (var i = 0; i < max(max_cells_in_block, 1); i = i + 1) {
            cell = update_cell_in_block(cell, block, ray, block_size);
            let packed = update_cubic_mip(volume_map, volume_sampler, volume_inv_dimensions, ray, cell, cubic_w, mip);
            cubic_w = packed.x;
            mip.value = packed.y;
            mip.distance = packed.z;
            mip.position = distance_to_position(ray, mip.distance);

            if (cell.terminated) {
                break;
            }
        }

        if (block.terminated) {
            break;
        }
    }

    mip.position = distance_to_position(ray, mip.distance);
    return mip;
}
