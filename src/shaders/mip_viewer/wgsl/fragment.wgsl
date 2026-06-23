fn mip_viewer_fragment(
    volume_map: texture_3d<f32>,
    distance_words: ptr<storage, array<u32>, read>,
    frag_coord: vec2<f32>,
    resolution: vec2<f32>,
    inv_projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_model: mat4x4<f32>,
    volume_dimensions: vec3<f32>,
    volume_inv_dimensions: vec3<f32>,
    distance_dimensions: vec3<f32>,
    ray_direction: vec3<f32>,
    ray_inv_direction: vec3<f32>,
    ray_sign_direction: vec3<f32>,
    ray_step_distances: vec3<f32>,
    ray_step_distance: f32,
    ray_dominant_axis: f32,
    ray_quadrant_index: f32,
    ray_group_index: f32,
    ray_reverse: f32,
    box_min_position: vec3<f32>,
    box_max_position: vec3<f32>,
    box_min_distance: f32,
    box_max_distance: f32,
    box_span_distance: f32,
    shading_colormap: f32,
    debug_option: f32,
    debug_enabled: f32,
    distance_variation: f32,
    marching_method: f32,
    skipping_method: f32,
    skipping_enabled: f32,
    block_size: f32,
    max_cells: f32,
    max_blocks: f32,
    max_cells_in_block: f32,
    max_traces: f32,
    max_traces_in_cell: f32,
    distance_words_per_voxel: f32
) -> vec4<f32>
{
    let volume_dimensions_i = vec3<i32>(round(volume_dimensions));
    let distance_dimensions_i = vec3<i32>(round(distance_dimensions));
    let ray_sign_direction_i = vec3<i32>(round(ray_sign_direction));
    let ray_dominant_axis_u = u32(round(max(ray_dominant_axis, 0.0)));
    let ray_quadrant_index_u = u32(round(max(ray_quadrant_index, 0.0)));
    let ray_group_index_u = u32(round(max(ray_group_index, 0.0)));
    let ray_reverse_i = i32(round(ray_reverse));
    let shading_colormap_i = i32(round(shading_colormap));
    let debug_option_i = i32(round(debug_option));
    let debug_enabled_i = i32(round(debug_enabled));
    let distance_variation_i = i32(round(distance_variation));
    let marching_method_i = i32(round(marching_method));
    let skipping_method_i = i32(round(skipping_method));
    let skipping_enabled_i = i32(round(skipping_enabled));
    let block_size_i = i32(round(block_size));
    let max_cells_i = i32(round(max(max_cells, 1.0)));
    let max_blocks_i = i32(round(max(max_blocks, 1.0)));
    let max_cells_in_block_i = i32(round(max(max_cells_in_block, 1.0)));
    let max_traces_in_cell_i = i32(round(max(max_traces_in_cell, 1.0)));
    let distance_words_per_voxel_u = u32(round(max(distance_words_per_voxel, 1.0)));

    var ray = make_ray(
        frag_coord,
        resolution,
        inv_projection,
        inv_view,
        inv_model,
        volume_dimensions_i,
        ray_direction,
        ray_inv_direction,
        ray_sign_direction_i,
        ray_step_distances,
        ray_step_distance,
        box_min_position,
        box_max_position
    );

    if (ray.discarded) {
        discard;
    }

    var mip = MipState(0.0, ray.start_distance, ray.start_position);

    if (marching_method_i == 0) {
        if (skipping_enabled_i == 1) {
            mip = march_cells_in_blocks(
                volume_map,
                distance_words,
                volume_inv_dimensions,
                distance_dimensions_i,
                ray,
                ray_dominant_axis_u,
                ray_quadrant_index_u,
                ray_group_index_u,
                distance_variation_i,
                skipping_method_i,
                block_size_i,
                max_blocks_i,
                max_cells_in_block_i,
                distance_words_per_voxel_u
            );
        } else {
            mip = march_cells_in_cells(
                volume_map,
                volume_inv_dimensions,
                ray,
                max_cells_i
            );
        }
    } else {
        if (skipping_enabled_i == 1) {
            mip = march_traces_in_cells_in_blocks(
                volume_map,
                distance_words,
                volume_inv_dimensions,
                distance_dimensions_i,
                ray,
                ray_dominant_axis_u,
                ray_quadrant_index_u,
                ray_group_index_u,
                distance_variation_i,
                skipping_method_i,
                block_size_i,
                max_blocks_i,
                max_cells_in_block_i,
                max_traces_in_cell_i,
                distance_words_per_voxel_u
            );
        } else {
            mip = march_traces_in_cells(
                volume_map,
                volume_inv_dimensions,
                ray,
                max_cells_i,
                max_traces_in_cell_i
            );
        }
    }

    let frag_color = colormap(shading_colormap_i, mip.value);
    var color = debug_color(
        debug_option_i,
        frag_color,
        distance_words,
        ray,
        mip,
        debug_enabled_i,
        volume_inv_dimensions,
        distance_dimensions_i,
        box_min_position,
        box_max_position,
        box_min_distance,
        box_max_distance,
        box_span_distance,
        ray_dominant_axis_u,
        ray_quadrant_index_u,
        ray_group_index_u,
        ray_reverse_i != 0,
        distance_variation_i,
        skipping_method_i,
        block_size_i,
        distance_words_per_voxel_u
    );

    if (ray_reverse_i != 0) {
        color = color;
    }

    return vec4<f32>(color, 1.0);
}

struct RayState
{
    discarded: bool,
    phase: f32,
    origin: vec3<f32>,
    direction: vec3<f32>,
    inv_direction: vec3<f32>,
    sign_direction: vec3<i32>,
    step_distances: vec3<f32>,
    step_distance: f32,
    eps_distance: f32,
    eps_direction: vec3<f32>,
    start_distance: f32,
    end_distance: f32,
    span_distance: f32,
    start_position: vec3<f32>,
    end_position: vec3<f32>,
};

struct CellState
{
    terminated: bool,
    coords: vec3<i32>,
    far_distances: vec3<f32>,
    entry_distance: f32,
    exit_distance: f32,
    entry_position: vec3<f32>,
    exit_position: vec3<f32>,
    exit_step: vec3<i32>,
};

struct BlockState
{
    empty: bool,
    prev_empty: bool,
    terminated: bool,
    coords: vec3<i32>,
    step_radius: i32,
    entry_distance: f32,
    exit_distance: f32,
    entry_position: vec3<f32>,
    exit_position: vec3<f32>,
    entry_step: vec3<i32>,
    exit_step: vec3<i32>,
};

struct MipState
{
    value: f32,
    distance: f32,
    position: vec3<f32>,
};

fn debug_color(
    option: i32,
    frag_color: vec3<f32>,
    distance_words: ptr<storage, array<u32>, read>,
    ray: RayState,
    mip: MipState,
    debug_enabled: i32,
    volume_inv_dimensions: vec3<f32>,
    distance_dimensions: vec3<i32>,
    box_min_position: vec3<f32>,
    box_max_position: vec3<f32>,
    box_min_distance: f32,
    box_max_distance: f32,
    box_span_distance: f32,
    ray_dominant_axis: u32,
    ray_quadrant_index: u32,
    ray_group_index: u32,
    ray_reverse: bool,
    distance_variation: i32,
    skipping_method: i32,
    block_size: i32,
    distance_words_per_voxel: u32
) -> vec3<f32>
{
    if (debug_enabled == 0 || option == 0 || option == 516) {
        return frag_color;
    }

    if (option == 101) {
        return vec3<f32>(select(0.0, 1.0, ray.discarded));
    }
    if (option == 102) {
        return ray.direction * 0.5 + vec3<f32>(0.5);
    }
    if (option == 103) {
        return vec3<f32>(ray.sign_direction) * 0.5 + vec3<f32>(0.5);
    }
    if (option == 104) {
        return vec3<f32>(clamp(ray.step_distance, 0.0, 1.0));
    }
    if (option == 105) {
        return vec3<f32>(clamp(map_range(box_min_distance, box_max_distance, ray.start_distance), 0.0, 1.0));
    }
    if (option == 106) {
        return vec3<f32>(clamp(map_range(box_min_distance, box_max_distance, ray.end_distance), 0.0, 1.0));
    }
    if (option == 107) {
        return vec3<f32>(clamp(map_range(0.0, max(box_span_distance, 1.0e-4), ray.span_distance), 0.0, 1.0));
    }
    if (option == 108) {
        return clamp(map_vec3(box_min_position, box_max_position, ray.start_position), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (option == 109) {
        return clamp(map_vec3(box_min_position, box_max_position, ray.end_position), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (option == 110) {
        return axis_color(ray_dominant_axis);
    }
    if (option == 111) {
        return indexed_gray(ray_quadrant_index, 4.0);
    }
    if (option == 112) {
        return axis_color(ray_group_index / 4u) * (1.0 - f32(ray_group_index % 4u) * 0.25);
    }
    if (option == 113) {
        return vec3<f32>(select(0.0, 1.0, ray_reverse));
    }
    if (option == 114) {
        return vec3<f32>(ray.phase);
    }

    if (option == 453) {
        return vec3<f32>(clamp(map_range(box_min_distance, box_max_distance, mip.distance), 0.0, 1.0));
    }
    if (option == 454) {
        return clamp(map_vec3(box_min_position, box_max_position, mip.position), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (option == 455) {
        return vec3<f32>(clamp(mip.value, 0.0, 1.0));
    }
    if (option == 451) {
        return vec3<f32>(select(0.0, 1.0, mip.distance >= ray.end_distance - ray.eps_distance));
    }
    if (option == 452) {
        return vec3<f32>(select(0.0, 1.0, mip.value > 0.0));
    }
    if (option == 456) {
        return vec3<f32>(0.5, 0.5, 1.0);
    }
    if (option == 457 || option == 458 || option == 459) {
        return vec3<f32>(clamp(mip.value, 0.0, 1.0));
    }

    let cell_coords = position_to_cell_coords(mip.position);
    let block_coords = position_to_block_coords(mip.position, block_size);
    let block_sample = sample_distance(
        distance_words,
        block_coords,
        distance_dimensions,
        ray_dominant_axis,
        ray_quadrant_index,
        ray_group_index,
        distance_variation,
        distance_words_per_voxel
    );
    let block_exit = intersect_block_exit(block_coords, select(1, block_sample.distance, skipping_method == 1), block_size, ray);
    let cell_exit = intersect_cell_far_distances(cell_coords_to_far_distances(cell_coords, ray));

    if (option == 401) {
        return vec3<f32>(clamp(f32(block_sample.distance) / 64.0, 0.0, 1.0));
    }
    if (option == 402) {
        return vec3<f32>(select(0.0, 1.0, block_sample.empty));
    }
    if (option == 403) {
        return vec3<f32>(select(0.0, 1.0, block_exit.distance > ray.end_distance - ray.eps_distance));
    }
    if (option == 404) {
        return clamp(vec3<f32>(block_coords) / vec3<f32>(max(distance_dimensions - vec3<i32>(1), vec3<i32>(1))), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (option == 405) {
        return vec3<f32>(block_exit.step) * 0.5 + vec3<f32>(0.5);
    }
    if (option == 406) {
        return vec3<f32>(block_exit.step) * 0.5 + vec3<f32>(0.5);
    }
    if (option == 407) {
        return vec3<f32>(clamp(map_range(box_min_distance, box_max_distance, mip.distance), 0.0, 1.0));
    }
    if (option == 408) {
        return vec3<f32>(clamp(map_range(box_min_distance, box_max_distance, block_exit.distance), 0.0, 1.0));
    }
    if (option == 409) {
        return vec3<f32>(clamp(map_range(0.0, max(box_span_distance, 1.0e-4), block_exit.distance - mip.distance), 0.0, 1.0));
    }

    if (option == 202) {
        return vec3<f32>(select(0.0, 1.0, cell_exit.distance > ray.end_distance - ray.eps_distance));
    }
    if (option == 204) {
        let dims = max(vec3<i32>(round(vec3<f32>(1.0) / max(volume_inv_dimensions, vec3<f32>(1.0e-8)))) - vec3<i32>(1), vec3<i32>(1));
        return clamp(vec3<f32>(cell_coords) / vec3<f32>(dims), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (option == 205) {
        return vec3<f32>(cell_exit.step) * 0.5 + vec3<f32>(0.5);
    }
    if (option == 206 || option == 303) {
        return vec3<f32>(clamp(map_range(box_min_distance, box_max_distance, mip.distance), 0.0, 1.0));
    }
    if (option == 207) {
        return vec3<f32>(clamp(map_range(box_min_distance, box_max_distance, cell_exit.distance), 0.0, 1.0));
    }
    if (option == 208) {
        return vec3<f32>(clamp(map_range(0.0, max(box_span_distance, 1.0e-4), cell_exit.distance - mip.distance), 0.0, 1.0));
    }
    if (option == 209) {
        return clamp(map_vec3(vec3<f32>(box_min_distance), vec3<f32>(box_max_distance), cell_coords_to_far_distances(cell_coords, ray)), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    if (option == 301) {
        return vec3<f32>(select(0.0, 1.0, mip.value > 0.0));
    }
    if (option == 302) {
        return vec3<f32>(select(0.0, 1.0, mip.distance >= ray.end_distance - ray.eps_distance));
    }
    if (option == 304) {
        return clamp(map_vec3(box_min_position, box_max_position, mip.position), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (option == 305) {
        return vec3<f32>(clamp(mip.value, 0.0, 1.0));
    }

    if (option == 511) {
        return frag_color;
    }

    if (option == 801) {
        return vec3<f32>(clamp(mip.value, 0.0, 1.0), clamp(map_range(box_min_distance, box_max_distance, mip.distance), 0.0, 1.0), 0.0);
    }
    if (option == 802 || option == 803) {
        return frag_color;
    }
    if (option == 804) {
        return clamp(map_vec3(box_min_position, box_max_position, mip.position), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (option == 805) {
        return vec3<f32>(clamp(mip.value, 0.0, 1.0));
    }

    if (option == 901) {
        return vec3<f32>(clamp(f32(max_cells_in_debug(block_size, distance_dimensions)) / 512.0, 0.0, 1.0));
    }
    if (option == 902) {
        return vec3<f32>(clamp(f32(max(block_size * 3 - 2, 1)) / 32.0, 0.0, 1.0));
    }
    if (option == 903 || option == 904) {
        return vec3<f32>(clamp(mip.value, 0.0, 1.0));
    }
    if (option == 905 || option == 906 || option == 907 || option == 908 || option == 909) {
        return vec3<f32>(clamp(f32(block_sample.distance) / 64.0, 0.0, 1.0));
    }

    if (option >= 1000 && option <= 1009) {
        return frag_color;
    }

    return frag_color;
}

struct DistanceSample
{
    distance: i32,
    empty: bool,
};

struct ExitHit
{
    distance: f32,
    step: vec3<i32>,
};

struct CubicMax
{
    value: f32,
    point: f32,
};

fn map_range(a: f32, b: f32, x: f32) -> f32
{
    return (x - a) / max(b - a, 1.0e-8);
}

fn map_vec3(a: vec3<f32>, b: vec3<f32>, x: vec3<f32>) -> vec3<f32>
{
    return (x - a) / max(b - a, vec3<f32>(1.0e-8));
}

fn axis_color(axis: u32) -> vec3<f32>
{
    if (axis == 0u) {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    if (axis == 1u) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn indexed_gray(index: u32, count: f32) -> vec3<f32>
{
    return vec3<f32>(1.0 - f32(index) / max(count - 1.0, 1.0));
}

fn max_cells_in_debug(block_size: i32, distance_dimensions: vec3<i32>) -> i32
{
    return max((distance_dimensions.x + distance_dimensions.y + distance_dimensions.z) * max(block_size, 1), 1);
}

fn random3(pos_in: vec3<f32>) -> f32
{
    var pos = fract(pos_in * vec3<f32>(443.897, 441.423, 0.0973));
    pos = pos + dot(pos, pos.zyx + vec3<f32>(31.32));
    return fract((pos.x + pos.y) * pos.z);
}

fn get_ray_origin(
    frag_coord: vec2<f32>,
    resolution: vec2<f32>,
    inv_projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_model: mat4x4<f32>,
    volume_dimensions: vec3<i32>
) -> vec3<f32>
{
    let gl_frag_coord = vec2<f32>(frag_coord.x, resolution.y - frag_coord.y);
    let uv = (gl_frag_coord + vec2<f32>(0.5)) / resolution;
    let ndc = uv * 2.0 - vec2<f32>(1.0);
    let clip_position = vec4<f32>(ndc, 0.0, 1.0);
    let view_position = inv_projection * clip_position;
    let world_position = inv_view * vec4<f32>(view_position.xyz / view_position.w, 1.0);
    let local_position = inv_model * world_position;
    return (local_position.xyz / local_position.w + vec3<f32>(0.5)) * vec3<f32>(volume_dimensions);
}

fn mmax3(v: vec3<f32>) -> f32
{
    return max(v.x, max(v.y, v.z));
}

fn mmin3(v: vec3<f32>) -> f32
{
    return min(v.x, min(v.y, v.z));
}

fn intersect_box(
    box_min: vec3<f32>,
    box_max: vec3<f32>,
    start: vec3<f32>,
    inv_dir: vec3<f32>
) -> vec2<f32>
{
    let b_min = (box_min - start) * inv_dir;
    let b_max = (box_max - start) * inv_dir;
    let t_entry = mmax3(min(b_min, b_max));
    let t_exit = mmin3(max(b_min, b_max));
    return vec2<f32>(t_entry, t_exit);
}

fn make_ray(
    frag_coord: vec2<f32>,
    resolution: vec2<f32>,
    inv_projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_model: mat4x4<f32>,
    volume_dimensions: vec3<i32>,
    direction: vec3<f32>,
    inv_direction: vec3<f32>,
    sign_direction: vec3<i32>,
    step_distances: vec3<f32>,
    step_distance: f32,
    box_min_position: vec3<f32>,
    box_max_position: vec3<f32>
) -> RayState
{
    let origin = get_ray_origin(frag_coord, resolution, inv_projection, inv_view, inv_model, volume_dimensions);
    let hit = intersect_box(box_min_position, box_max_position, origin, inv_direction);
    var ray = RayState(
        hit.y <= hit.x,
        random3(origin),
        origin,
        direction,
        inv_direction,
        sign_direction,
        step_distances,
        step_distance,
        step_distance * 0.001,
        direction * 0.001,
        hit.x,
        hit.y,
        hit.y - hit.x,
        origin + direction * hit.x,
        origin + direction * hit.y
    );

    return ray;
}

fn distance_to_position(ray: RayState, t: f32) -> vec3<f32>
{
    return ray.origin + ray.direction * t;
}

fn sample_volume(
    volume_map: texture_3d<f32>,
    position: vec3<f32>,
    volume_inv_dimensions: vec3<f32>
) -> f32
{
    let texture_position = clamp(position * volume_inv_dimensions, vec3<f32>(0.0), vec3<f32>(1.0));
    return textureSample(volume_map, volume_map_sampler, texture_position).r;
}

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
    volume_inv_dimensions: vec3<f32>,
    ray: RayState,
    max_traces_in_cell: i32
) -> MipState
{
    let trace_step = ray.step_distance / f32(max(max_traces_in_cell - 1, 1));
    let trace_distance = snap_trace_distance_ceil(ray.start_distance, trace_step, ray.phase);
    let trace_position = distance_to_position(ray, trace_distance);
    let trace_value = sample_volume(volume_map, trace_position, volume_inv_dimensions);
    return MipState(trace_value, trace_distance, trace_position);
}

fn march_traces_in_cells(
    volume_map: texture_3d<f32>,
    volume_inv_dimensions: vec3<f32>,
    ray: RayState,
    max_cells: i32,
    max_traces_in_cell: i32
) -> MipState
{
    var cell = start_cell_in_ray(ray);
    var mip = make_initial_trace_mip(volume_map, volume_inv_dimensions, ray, max_traces_in_cell);

    for (var j = 0; j < max(max_cells, 1); j = j + 1) {
        cell = update_cell_in_ray(cell, ray);

        for (var i = 1; i < max(max_traces_in_cell, 1); i = i + 1) {
            let alpha = f32(i) / f32(max(max_traces_in_cell - 1, 1));
            let trace_distance = mix(cell.entry_distance, cell.exit_distance, alpha);
            let trace_position = distance_to_position(ray, trace_distance);
            let trace_value = sample_volume(volume_map, trace_position, volume_inv_dimensions);
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
    var mip = make_initial_trace_mip(volume_map, volume_inv_dimensions, ray, max_traces_in_cell);

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
                let trace_value = sample_volume(volume_map, trace_position, volume_inv_dimensions);
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
        sample_volume(volume_map, cell.entry_position + span_vector * (1.0 / 3.0), volume_inv_dimensions),
        sample_volume(volume_map, cell.entry_position + span_vector * (2.0 / 3.0), volume_inv_dimensions),
        sample_volume(volume_map, cell.exit_position, volume_inv_dimensions)
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
    volume_inv_dimensions: vec3<f32>,
    ray: RayState,
    max_cells: i32
) -> MipState
{
    var cell = start_cell_in_ray(ray);
    var cubic_w = sample_volume(volume_map, ray.start_position, volume_inv_dimensions);
    var mip = MipState(cubic_w, ray.start_distance, ray.start_position);

    for (var i = 0; i < max(max_cells, 1); i = i + 1) {
        cell = update_cell_in_ray(cell, ray);
        let packed = update_cubic_mip(volume_map, volume_inv_dimensions, ray, cell, cubic_w, mip);
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
    var cubic_w = sample_volume(volume_map, ray.start_position, volume_inv_dimensions);
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
            cubic_w = sample_volume(volume_map, block.entry_position, volume_inv_dimensions);
        }

        var cell = start_cell_in_block(block, ray, block_size);

        for (var i = 0; i < max(max_cells_in_block, 1); i = i + 1) {
            cell = update_cell_in_block(cell, block, ray, block_size);
            let packed = update_cubic_mip(volume_map, volume_inv_dimensions, ray, cell, cubic_w, mip);
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

fn get_u4(v: vec4<u32>, i: u32) -> u32
{
    if (i == 0u) {
        return v.x;
    }
    if (i == 1u) {
        return v.y;
    }
    if (i == 2u) {
        return v.z;
    }
    return v.w;
}

fn load_distance_words(
    distance_words: ptr<storage, array<u32>, read>,
    coords: vec3<i32>,
    dimensions: vec3<i32>,
    words_per_voxel: u32
) -> vec4<u32>
{
    if (
        any(coords < vec3<i32>(0)) ||
        any(coords >= dimensions) ||
        any(dimensions <= vec3<i32>(0))
    ) {
        return vec4<u32>(0u);
    }

    let linear = (
        (u32(coords.z) * u32(dimensions.y) + u32(coords.y)) *
        u32(dimensions.x) + u32(coords.x)
    ) * words_per_voxel;

    var packed = vec4<u32>(0u);

    if (words_per_voxel > 0u) {
        packed.x = (*distance_words)[linear + 0u];
    }
    if (words_per_voxel > 1u) {
        packed.y = (*distance_words)[linear + 1u];
    }
    if (words_per_voxel > 2u) {
        packed.z = (*distance_words)[linear + 2u];
    }
    if (words_per_voxel > 3u) {
        packed.w = (*distance_words)[linear + 3u];
    }

    return packed;
}

fn sample_distance(
    distance_words: ptr<storage, array<u32>, read>,
    coords: vec3<i32>,
    dimensions: vec3<i32>,
    dominant_axis: u32,
    quadrant_index: u32,
    group_index: u32,
    distance_variation: i32,
    words_per_voxel: u32
) -> DistanceSample
{
    let u = load_distance_words(distance_words, coords, dimensions, words_per_voxel);
    var packed = 0u;
    var shift = 0u;
    var mask = 1u;

    if (distance_variation == 0) {
        packed = u.x;
        shift = group_index;
        mask = 0x1u;
    } else if (distance_variation == 1) {
        packed = get_u4(u, quadrant_index);
        shift = dominant_axis * 5u;
        mask = select(0x3fu, 0x1fu, dominant_axis < 2u);
    } else if (distance_variation == 2) {
        packed = get_u4(u, dominant_axis);
        shift = quadrant_index * 8u;
        mask = 0xffu;
    } else {
        packed = get_u4(u, quadrant_index);
        shift = dominant_axis * 11u;
        mask = select(0x3ffu, 0x7ffu, dominant_axis < 2u);
    }

    let d = (packed >> shift) & mask;
    return DistanceSample(i32(max(d, 1u)), d != 0u);
}

fn palette4(
    t: f32,
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    d: vec3<f32>
) -> vec3<f32>
{
    return a + b * cos(6.283185 * (c * t + d));
}

fn palette8(
    t: f32,
    a: vec3<f32>,
    b0: vec3<f32>,
    c0: vec3<f32>,
    d0: vec3<f32>,
    b1: vec3<f32>,
    c1: vec3<f32>,
    d1: vec3<f32>
) -> vec3<f32>
{
    return a + b0 * cos(6.283185 * (c0 * t + d0)) + b1 * cos(6.283185 * (c1 * t + d1));
}

fn parula(t: f32) -> vec3<f32>
{
    return palette8(clamp(t, 0.0, 1.0), vec3<f32>(0.541454, 1.968902, 0.559818), vec3<f32>(0.460347, 1.998354, 0.429174), vec3<f32>(0.645892, 0.330375, 0.763244), vec3<f32>(0.309938, 0.387593, -0.187696), vec3<f32>(-0.154889, 0.704319, 0.011011), vec3<f32>(1.667578, 0.717541, 2.0), vec3<f32>(0.135293, 0.686891, 0.102776));
}

fn viridis(t: f32) -> vec3<f32>
{
    return palette8(clamp(t, 0.0, 1.0), vec3<f32>(0.425268, -0.364758, 0.418135), vec3<f32>(1.1258, 1.306212, -1.205918), vec3<f32>(0.778337, 0.16538, 1.148509), vec3<f32>(0.296775, 0.795553, 0.04792), vec3<f32>(0.937521, 0.012337, -1.069593), vec3<f32>(0.89106, 1.453194, 1.217589), vec3<f32>(0.781272, 0.775696, 0.520518));
}

fn turbo(t: f32) -> vec3<f32>
{
    return palette8(clamp(t, 0.0, 1.0), vec3<f32>(-1.173583, 1.089549, 0.363003), vec3<f32>(1.999787, -1.091536, 0.59681), vec3<f32>(0.168544, 0.690975, 1.090372), vec3<f32>(0.84329, 0.188794, -0.342117), vec3<f32>(0.265151, 1.180907, -0.376009), vec3<f32>(1.460237, 0.38146, 1.554106), vec3<f32>(0.9959, 0.331064, 0.346513));
}

fn hsv(t: f32) -> vec3<f32>
{
    return palette8(clamp(t, 0.0, 1.0), vec3<f32>(0.34593, -1.151078, -1.32791), vec3<f32>(0.703838, 1.816049, 1.999886), vec3<f32>(1.417315, 0.237319, 0.228754), vec3<f32>(-0.210907, -0.12051, 0.880709), vec3<f32>(0.312792, 0.543745, 0.517213), vec3<f32>(2.0, 0.995237, 1.0185), vec3<f32>(-0.003169, 0.719422, 0.269492));
}

fn hot(t: f32) -> vec3<f32>
{
    return palette8(clamp(t, 0.0, 1.0), vec3<f32>(0.434751, 0.499201, 0.553319), vec3<f32>(0.117871, 0.59026, 1.998983), vec3<f32>(1.395636, 0.560375, 0.692793), vec3<f32>(0.516179, 0.436002, 0.254912), vec3<f32>(0.669133, -0.097445, 1.467678), vec3<f32>(0.457411, 1.67373, 0.848067), vec3<f32>(0.679382, 0.311157, 0.700754));
}

fn cool(t: f32) -> vec3<f32>
{
    return palette4(clamp(t, 0.0, 1.0), vec3<f32>(0.500001, 0.5, 1.0), vec3<f32>(1.999655, 1.999654, 0.0), vec3<f32>(0.080099, 0.080099, 0.745176), vec3<f32>(0.70995, 0.20995, 0.127412));
}

fn spring(t: f32) -> vec3<f32>
{
    return palette4(clamp(t, 0.0, 1.0), vec3<f32>(1.0, 0.500001, 0.500001), vec3<f32>(0.0, 1.999808, 1.999812), vec3<f32>(0.745176, 0.080093, 0.080093), vec3<f32>(0.127412, 0.709953, 0.209954));
}

fn summer(t: f32) -> vec3<f32>
{
    return palette4(clamp(t, 0.0, 1.0), vec3<f32>(0.499979, 0.749503, 0.4), vec3<f32>(1.998634, 1.447229, 0.0), vec3<f32>(0.080141, 0.055144, 0.73234), vec3<f32>(0.709931, 0.722482, 0.13383));
}

fn autumn(t: f32) -> vec3<f32>
{
    return palette4(clamp(t, 0.0, 1.0), vec3<f32>(1.0, 0.5, 0.0), vec3<f32>(0.0, 1.999988, 0.0), vec3<f32>(0.745176, 0.080086, 0.714514), vec3<f32>(0.127412, 0.709957, 0.142743));
}

fn winter(t: f32) -> vec3<f32>
{
    return palette4(clamp(t, 0.0, 1.0), vec3<f32>(0.0, 0.500001, 0.75001), vec3<f32>(0.0, 1.999953, 1.430454), vec3<f32>(0.714514, 0.080087, 0.055794), vec3<f32>(0.142743, 0.709956, 0.222104));
}

fn gray(t: f32) -> vec3<f32>
{
    return palette4(clamp(t, 0.0, 1.0), vec3<f32>(0.5), vec3<f32>(1.999993), vec3<f32>(0.080086), vec3<f32>(0.709957));
}

fn bone(t: f32) -> vec3<f32>
{
    return palette4(clamp(t, 0.0, 1.0), vec3<f32>(1.72856, 0.535898, -0.726993), vec3<f32>(2.0, 0.583406, 2.0), vec3<f32>(0.10069, 0.318709, 0.102955), vec3<f32>(0.587366, 0.576509, 0.809972));
}

fn copper(t: f32) -> vec3<f32>
{
    return palette4(clamp(t, 0.0, 1.0), vec3<f32>(0.475474, 0.390606, 0.248753), vec3<f32>(0.545609, 1.863885, 1.362622), vec3<f32>(0.404723, 0.067001, 0.058301), vec3<f32>(0.59671, 0.716499, 0.720849));
}

fn pink(t: f32) -> vec3<f32>
{
    return palette8(clamp(t, 0.0, 1.0), vec3<f32>(-0.483794, -0.77793, -0.293682), vec3<f32>(1.988198, 1.900866, 1.999968), vec3<f32>(0.313872, 0.125783, 0.246824), vec3<f32>(-0.242539, 0.825271, 0.729959), vec3<f32>(0.593545, 0.029151, 0.73639), vec3<f32>(0.58351, 1.77675, 0.478608), vec3<f32>(1.077824, 0.775436, 1.088446));
}

fn jet(t: f32) -> vec3<f32>
{
    return palette8(clamp(t, 0.0, 1.0), vec3<f32>(0.276022, 0.527041, 0.353694), vec3<f32>(0.597922, -0.201586, 0.559465), vec3<f32>(0.512799, 1.045576, 0.596704), vec3<f32>(0.573475, -0.103501, -0.104961), vec3<f32>(0.238268, 0.399577, -0.201583), vec3<f32>(1.424259, 1.047211, 1.500953), vec3<f32>(-0.01231, 0.51882, 0.058936));
}

fn pasteljet(t: f32) -> vec3<f32>
{
    return palette8(clamp(t, 0.0, 1.0), vec3<f32>(0.677942, 0.350764, 0.469527), vec3<f32>(-0.043513, -0.050359, 0.266445), vec3<f32>(2.0, 2.0, 0.698727), vec3<f32>(0.497311, 0.518843, -0.207068), vec3<f32>(0.338745, -0.561075, 0.062141), vec3<f32>(0.863745, 0.650179, 2.0), vec3<f32>(0.43731, 0.223441, 0.01022));
}

fn plasma(t: f32) -> vec3<f32>
{
    return palette8(clamp(t, 0.0, 1.0), vec3<f32>(0.260353, 0.643741, 0.401505), vec3<f32>(0.925001, 0.6413, 0.244923), vec3<f32>(0.451398, 0.362059, 0.688036), vec3<f32>(0.674213, 0.466824, -0.150012), vec3<f32>(0.24903, 0.020714, 0.015128), vec3<f32>(0.709559, 1.885207, 2.0), vec3<f32>(1.05819, 0.031753, 0.567507));
}

fn inferno(t: f32) -> vec3<f32>
{
    return palette8(clamp(t, 0.0, 1.0), vec3<f32>(-0.575419, 0.41734, 0.285807), vec3<f32>(1.543616, 1.325364, 2.0), vec3<f32>(0.174341, 0.656644, 1.696938), vec3<f32>(0.815911, 0.396624, -0.608185), vec3<f32>(0.095489, 0.951299, 1.871821), vec3<f32>(0.907917, 0.757037, 1.749972), vec3<f32>(0.380506, 0.870792, -0.130654));
}

fn magma(t: f32) -> vec3<f32>
{
    return palette8(clamp(t, 0.0, 1.0), vec3<f32>(-0.386429, 0.394408, 0.40972), vec3<f32>(1.414174, 1.143924, 1.416815), vec3<f32>(0.189647, 0.67694, 0.641115), vec3<f32>(0.79347, 0.38845, -0.56849), vec3<f32>(0.079624, 0.813267, 1.330803), vec3<f32>(1.052351, 0.800573, 0.764647), vec3<f32>(0.286256, 0.854038, 0.863334));
}

fn cividis(t: f32) -> vec3<f32>
{
    return palette8(clamp(t, 0.0, 1.0), vec3<f32>(0.471158, 0.89498, -1.390197), vec3<f32>(1.441757, 1.133394, 1.864691), vec3<f32>(0.670921, 0.196915, 0.145757), vec3<f32>(0.424741, 0.578773, -0.064678), vec3<f32>(1.085652, 0.238494, 0.032514), vec3<f32>(0.768613, 0.378699, 1.46755), vec3<f32>(0.876007, 0.999533, -0.060326));
}

fn colormap(kind: i32, t: f32) -> vec3<f32>
{
    if (kind == 0) {
        return parula(t);
    } else if (kind == 1) {
        return turbo(t);
    } else if (kind == 2) {
        return hsv(t);
    } else if (kind == 3) {
        return hot(t);
    } else if (kind == 4) {
        return cool(t);
    } else if (kind == 5) {
        return spring(t);
    } else if (kind == 6) {
        return summer(t);
    } else if (kind == 7) {
        return autumn(t);
    } else if (kind == 8) {
        return winter(t);
    } else if (kind == 9) {
        return gray(t);
    } else if (kind == 10) {
        return bone(t);
    } else if (kind == 11) {
        return copper(t);
    } else if (kind == 12) {
        return pink(t);
    } else if (kind == 13) {
        return jet(t);
    } else if (kind == 14) {
        return pasteljet(t);
    } else if (kind == 15) {
        return viridis(t);
    } else if (kind == 16) {
        return plasma(t);
    } else if (kind == 17) {
        return inferno(t);
    } else if (kind == 18) {
        return magma(t);
    } else if (kind == 19) {
        return cividis(t);
    }

    return viridis(t);
}
