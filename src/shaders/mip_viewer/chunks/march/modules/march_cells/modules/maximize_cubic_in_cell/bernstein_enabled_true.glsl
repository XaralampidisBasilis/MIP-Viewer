
// Cull with Bernstein coefficients before the full cubic maximize step.
cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;
cubic.maximize = any(greaterThan(cubic.bernstein_coeffs, vec4(mip.value)));

mip.update = false;

if (cubic.maximize)
{
    // MAXIMIZE_CUBIC
    cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
    CubicMax cubic_max = cubicMaxOnUnitInterval(cubic.coeffs, cubic.values.x, cubic.values.w);

    cubic.max_value = cubic_max.value;
    cubic.argmax_point = cubic_max.point;

    mip.update = cubic.max_value > mip.value;

    #if DEBUG_ENABLED == 1

        stats.num_maxima += 1;

    #endif

}
