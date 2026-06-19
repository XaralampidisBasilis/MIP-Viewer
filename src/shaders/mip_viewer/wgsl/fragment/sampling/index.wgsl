fn sample_volume(
    volume_map: ptr<storage, array<u32>, read>,
    position: vec3<f32>,
    volume_inv_dimensions: vec3<f32>
) -> f32
{
    let dims = max(vec3<i32>(round(vec3<f32>(1.0) / max(volume_inv_dimensions, vec3<f32>(1.0e-8)))), vec3<i32>(1));
    let p = clamp(position - vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(dims - vec3<i32>(1)));
    let p0 = vec3<i32>(floor(p));
    let p1 = min(p0 + vec3<i32>(1), dims - vec3<i32>(1));
    let f = fract(p);

    let c000 = load_volume_value(volume_map, p0, dims);
    let c100 = load_volume_value(volume_map, vec3<i32>(p1.x, p0.y, p0.z), dims);
    let c010 = load_volume_value(volume_map, vec3<i32>(p0.x, p1.y, p0.z), dims);
    let c110 = load_volume_value(volume_map, vec3<i32>(p1.x, p1.y, p0.z), dims);
    let c001 = load_volume_value(volume_map, vec3<i32>(p0.x, p0.y, p1.z), dims);
    let c101 = load_volume_value(volume_map, vec3<i32>(p1.x, p0.y, p1.z), dims);
    let c011 = load_volume_value(volume_map, vec3<i32>(p0.x, p1.y, p1.z), dims);
    let c111 = load_volume_value(volume_map, p1, dims);

    let c00 = mix(c000, c100, f.x);
    let c10 = mix(c010, c110, f.x);
    let c01 = mix(c001, c101, f.x);
    let c11 = mix(c011, c111, f.x);
    return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

fn load_volume_value(
    volume_map: ptr<storage, array<u32>, read>,
    coords: vec3<i32>,
    dims: vec3<i32>
) -> f32
{
    let c = clamp(coords, vec3<i32>(0), dims - vec3<i32>(1));
    let idx = u32((c.z * dims.y + c.y) * dims.x + c.x);
    let word = (*volume_map)[idx >> 1u];
    let pair = unpack2x16float(word);
    return select(pair.x, pair.y, (idx & 1u) == 1u);
}
