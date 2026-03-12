out vec3 v_position;
out vec3 v_camera_position;
out vec3 v_camera_direction;

flat out vec3  v_ray_direction;
flat out vec3  v_ray_inv_direction;
flat out float v_ray_spacing;
flat out ivec3 v_ray_signs;
flat out uint  v_ray_axis;
flat out uint  v_ray_idx;
flat out uint  v_ray_map;
flat out uint  v_ray_reverse;

#include "./chunks/uniforms/uniforms_volume"
#include "./chunks/utils/math/sum"
#include "./chunks/utils/math/ssign"
#include "./chunks/utils/math/argmax"

vec3 local_position_to_index_space(vec3 p)
{
    return (p + 0.5) * vec3(u_volume.dimensions);
}

vec3 local_vector_to_index_space(vec3 v)
{
    return v * vec3(u_volume.dimensions);
}

// Returns the 2-bit quadrant index (0..3) after projecting the ray sign
// configuration onto the plane orthogonal to the dominant axis.
//
// axis = 0 -> classify in the yz plane
// axis = 1 -> classify in the xz plane
// axis = 2 -> classify in the xy plane
uint ray_quad_idx(ivec3 signs, uint axis)
{
    uvec3 b = uvec3(greaterThan(signs, ivec3(0)));
    
    uint xy = b.x ^ b.y;
    uint xz = b.x ^ b.z;
    uint yz = b.y ^ b.z;

    if (axis == 0u) return (xz << 1) | xy; 
    if (axis == 1u) return (yz << 1) | xy; 
    if (axis == 2u) return (xz << 1) | yz;   
}

void main() 
{
    vec3 volume_dimensions = vec3(u_volume.dimensions);

    // Unit cube vertex in voxel/index space.
    v_position = local_position_to_index_space(position);

    // Camera position in local space, then in voxel/index space.
    vec3 camera_position_local = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
    vec3 camera_plane_position = local_position_to_index_space(camera_position_local);

    // Orthographic view direction in local space.
    // In view space, the camera looks down -Z.
    vec3 camera_direction_view = vec3(0.0, 0.0, -1.0);
    vec3 camera_direction_local = (inverse(modelViewMatrix) * vec4(camera_direction_view, 0.0)).xyz;

    // Scale into voxel space so direction respects non-uniform volume dimensions.
    v_camera_direction = local_vector_to_index_space(camera_direction_local);
    v_ray_direction = normalize(v_camera_direction);
    v_ray_inv_direction = 1.0 / v_ray_direction;

    // Compute ray spacing
    vec3 ray_abs_direction = abs(v_ray_direction);
    v_ray_signs = ivec3(ssign(v_ray_direction));
    v_ray_spacing = 1.0 / sum(ray_abs_direction);

    // Compute ray classifications
    v_ray_axis = uint(argmax(ray_abs_direction));
    v_ray_idx = ray_quad_idx(v_ray_signs, v_ray_axis);
    v_ray_map = v_ray_idx + v_ray_axis * 4u;

    // If we have negative sign in the ray direction in the dominant axis reverse the ray
    v_ray_reverse = uint(v_ray_signs[v_ray_axis] < 0);

    // For orthographic projection, each ray starts on the camera plane and
    // passes through the current vertex-aligned ray line.
    float t = dot(v_position - camera_plane_position, v_ray_direction);
    v_camera_position = v_position - v_ray_direction * t;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}