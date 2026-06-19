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
    let clip_position = vec4<f32>(ndc, -1.0, 1.0);
    let view_position = inv_projection * clip_position;
    let world_position = inv_view * vec4<f32>(view_position.xyz, 1.0);
    let local_position = inv_model * world_position;
    return (local_position.xyz + vec3<f32>(0.5)) * vec3<f32>(volume_dimensions);
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
