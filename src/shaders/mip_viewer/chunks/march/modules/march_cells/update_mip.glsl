

vec3 derivative = vec3(cubic.coeffs.y, 2.0 * cubic.coeffs.z, 2.0 * cubic.coeffs.w);

cubic.crit_points = quadratic_roots(derivative);
cubic.crit_points = clamp(cubic.crit_points, 0.0, 1.0);

vec2 ext_values = eval_poly(cubic.coeffs, cubic.crit_points);

cubic.max_value = mmax(cubic.values[0], cubic.values[3], ext_values[0], ext_values[1]);

if (cubic.max_value > mip.value)
{
    mip.value = cubic.max_value;
}

