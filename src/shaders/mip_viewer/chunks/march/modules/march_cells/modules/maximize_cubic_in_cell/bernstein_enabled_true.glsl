
// Cull with Bernstein coefficients before the full cubic maximize step.
cubic.bernstein_coeffs = cubic.values * CUBIC_INV_BERNSTEIN;
cubic.maximize = any(greaterThan(cubic.bernstein_coeffs, vec4(mip.value)));

mip.update = false;

if (cubic.maximize)
{
    // MAXIMIZE_CUBIC
    #include "../maximize_cubic_in_cell/bernstein_enabled_false"
}
