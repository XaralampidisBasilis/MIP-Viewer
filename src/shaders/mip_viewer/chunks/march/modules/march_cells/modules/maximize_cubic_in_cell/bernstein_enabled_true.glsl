
// Cull with Bernstein coefficients before the full cubic maximize step.
cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;

mip.update = 
    cubic.bernstein_coeffs.x > mip.value || 
    cubic.bernstein_coeffs.y > mip.value || 
    cubic.bernstein_coeffs.z > mip.value || 
    cubic.bernstein_coeffs.w > mip.value;

if (mip.update)
{
    // MAXIMIZE_CUBIC
    cubic.coeffs = cubic.values * CUBIC_INV_VANDER;
    CubicMax cubic_max = cubicMaxOnUnitInterval(cubic.coeffs, cubic.values.x, cubic.values.w);

    cubic.max_value = cubic_max.v;
    cubic.argmax_t = cubic_max.t;

    mip.update = cubic.max_value > mip.value;

    #if DEBUG_ENABLED == 1

        stats.num_cubics += 1;

    #endif

}
