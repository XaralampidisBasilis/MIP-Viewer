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
