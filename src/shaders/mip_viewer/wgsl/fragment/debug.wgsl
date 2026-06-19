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

