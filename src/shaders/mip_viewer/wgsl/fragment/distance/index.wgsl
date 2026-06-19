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
