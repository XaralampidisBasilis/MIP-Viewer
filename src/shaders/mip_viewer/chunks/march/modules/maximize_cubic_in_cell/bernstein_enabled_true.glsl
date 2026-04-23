
// Cull with Bernstein coefficients before the full cubic maximize step.
cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;
cubic.maximize = shouldMaximizeCubic(cubic.bernstein_coeffs, mip.value);

if (cubic.maximize)
{
    // MAXIMIZE_CUBIC
    cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
    CubicMax cubic_max = cubicMaxOnUnitInterval(cubic.coeffs, cubic.values.x, cubic.values.w);

    cubic.max_value = cubic_max.value;
    cubic.argmax_point = cubic_max.point;

    #if DEBUG_ENABLED == 1

        stats.num_maxima += 1;

    #endif
}
