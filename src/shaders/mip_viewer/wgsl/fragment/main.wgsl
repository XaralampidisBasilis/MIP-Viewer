fn mip_viewer_fragment(
    volume_map: ptr<storage, array<u32>, read>,
    distance_words: ptr<storage, array<u32>, read>,
    frag_coord: vec2<f32>,
    resolution: vec2<f32>,
    inv_projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_model: mat4x4<f32>,
    volume_dimensions: vec3<i32>,
    volume_inv_dimensions: vec3<f32>,
    distance_dimensions: vec3<i32>,
    ray_direction: vec3<f32>,
    ray_inv_direction: vec3<f32>,
    ray_sign_direction: vec3<i32>,
    ray_step_distances: vec3<f32>,
    ray_step_distance: f32,
    ray_dominant_axis: u32,
    ray_quadrant_index: u32,
    ray_group_index: u32,
    ray_reverse: bool,
    box_min_position: vec3<f32>,
    box_max_position: vec3<f32>,
    box_min_distance: f32,
    box_max_distance: f32,
    box_span_distance: f32,
    shading_colormap: i32,
    debug_option: i32,
    debug_enabled: i32,
    distance_variation: i32,
    marching_method: i32,
    skipping_method: i32,
    skipping_enabled: i32,
    block_size: i32,
    max_cells: i32,
    max_blocks: i32,
    max_cells_in_block: i32,
    max_traces: i32,
    max_traces_in_cell: i32,
    distance_words_per_voxel: u32
) -> vec4<f32>
{
    var ray = make_ray(
        frag_coord,
        resolution,
        inv_projection,
        inv_view,
        inv_model,
        volume_dimensions,
        ray_direction,
        ray_inv_direction,
        ray_sign_direction,
        ray_step_distances,
        ray_step_distance,
        box_min_position,
        box_max_position
    );

    if (ray.discarded) {
        discard;
    }

    var mip = MipState(0.0, ray.start_distance, ray.start_position);

    if (marching_method == 0) {
        if (skipping_enabled == 1) {
            mip = march_cells_in_blocks(
                volume_map,
                distance_words,
                volume_inv_dimensions,
                distance_dimensions,
                ray,
                ray_dominant_axis,
                ray_quadrant_index,
                ray_group_index,
                distance_variation,
                skipping_method,
                block_size,
                max_blocks,
                max_cells_in_block,
                distance_words_per_voxel
            );
        } else {
            mip = march_cells_in_cells(
                volume_map,
                volume_inv_dimensions,
                ray,
                max_cells
            );
        }
    } else {
        if (skipping_enabled == 1) {
            mip = march_traces_in_cells_in_blocks(
                volume_map,
                distance_words,
                volume_inv_dimensions,
                distance_dimensions,
                ray,
                ray_dominant_axis,
                ray_quadrant_index,
                ray_group_index,
                distance_variation,
                skipping_method,
                block_size,
                max_blocks,
                max_cells_in_block,
                max_traces_in_cell,
                distance_words_per_voxel
            );
        } else {
            mip = march_traces_in_cells(
                volume_map,
                volume_inv_dimensions,
                ray,
                max_cells,
                max_traces_in_cell
            );
        }
    }

    let frag_color = colormap(shading_colormap, mip.value);
    var color = debug_color(
        debug_option,
        frag_color,
        distance_words,
        ray,
        mip,
        debug_enabled,
        volume_inv_dimensions,
        distance_dimensions,
        box_min_position,
        box_max_position,
        box_min_distance,
        box_max_distance,
        box_span_distance,
        ray_dominant_axis,
        ray_quadrant_index,
        ray_group_index,
        ray_reverse,
        distance_variation,
        skipping_method,
        block_size,
        distance_words_per_voxel
    );

    if (ray_reverse) {
        color = color;
    }

    return vec4<f32>(color, 1.0);
}

