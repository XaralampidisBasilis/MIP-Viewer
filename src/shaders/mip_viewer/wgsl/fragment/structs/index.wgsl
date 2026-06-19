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
