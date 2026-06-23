fn mip_viewer_fragment(
    volume_map: texture_3d<f32>,
    volume_sampler: sampler,
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
    let max_traces_i = i32(round(max(max_traces, 1.0)));
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
    var stats = StatsState(0, 0, 0, 0, 0, 0, 0, 0, 0);

    if (marching_method_i == 0) {
        if (skipping_enabled_i == 1) {
            mip = march_cells_in_blocks(
                volume_map,
                volume_sampler,
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
                distance_words_per_voxel_u,
                &stats
            );
        } else {
            mip = march_cells_in_cells(
                volume_map,
                volume_sampler,
                volume_inv_dimensions,
                ray,
                max_cells_i,
                &stats
            );
        }
    } else {
        if (skipping_enabled_i == 1) {
            mip = march_traces_in_cells_in_blocks(
                volume_map,
                volume_sampler,
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
                distance_words_per_voxel_u,
                &stats
            );
        } else {
            mip = march_traces_in_cells(
                volume_map,
                volume_sampler,
                volume_inv_dimensions,
                ray,
                max_cells_i,
                max_traces_in_cell_i,
                &stats
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
        max_cells_i,
        max_blocks_i,
        max_traces_i,
        stats,
        distance_words_per_voxel_u
    );

    if (ray_reverse_i != 0) {
        color = color;
    }

    return vec4<f32>(color, 1.0);
}

