fn sample_volume(
    volume_map: texture_3d<f32>,
    volume_sampler: sampler,
    position: vec3<f32>,
    volume_inv_dimensions: vec3<f32>
) -> f32
{
    let texture_position = clamp(position * volume_inv_dimensions, vec3<f32>(0.0), vec3<f32>(1.0));
    return textureSample(volume_map, volume_sampler, texture_position).r;
}
