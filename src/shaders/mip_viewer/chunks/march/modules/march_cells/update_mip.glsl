

vec3 quadratic_coeffs = vec3(cubic.coeffs.y, 2.0 * cubic.coeffs.z, 2.0 * cubic.coeffs.w);

vec2 critical_points = quadratic_roots(quadratic_coeffs);
critical_points = clamp(critical_points, 0.0, 1.0);

vec2 extrema_values = eval_poly(cubic.coeffs, critical_points);

vec4 candidate_values = vec4(cubic.values[0], extrema_values, cubic.values[3]);
vec4 candidate_points = vec4(0.0, critical_points, 1.0);

int i_max = argmax(candidate_values);
float max_value = candidate_values[i_max];
float argmax_point = candidate_points[i_max];

if (max_value > mip.value)
{
    mip.value = max_value;
    mip.distance = mix(cell.entry_distance, cell.exit_distance, argmax_point);
}

