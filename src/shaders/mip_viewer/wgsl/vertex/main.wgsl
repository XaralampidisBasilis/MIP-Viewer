fn mip_viewer_vertex(position: vec3<f32>, model_view_projection: mat4x4<f32>) -> vec4<f32>
{
    return model_view_projection * vec4<f32>(position, 1.0);
}

