#ifndef TRIQUADRATIC_BSPLINE_FILTER
#define TRIQUADRATIC_BSPLINE_FILTER

/* Sources
One Step Further Beyond Trilinear Interpolation and Central
Differences: Triquadratic Reconstruction and its Analytic
Derivatives at the Cost of One Additional Texture Fetch 
(https://onlinelibrary.wiley.com/doi/10.1111/cgf.14753),

GPU Gems 2, Chapter 20. Fast Third-Order Texture Filtering 
(https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering),
*/

// Triquadratic B-spline interpolation basis

void triquadratic_bspline_basis(in sampler3D tex, in vec3 coords, out vec3 p0, out vec3 p1, out vec3 g0)
{
    // Get size and normalized step
    vec3 size = vec3(textureSize(tex, 0));
    vec3 t = 1.0 / size;

    // Convert to voxel-space and compute local coordinates
    vec3 x = coords - 0.5;
    vec3 b = x - round(x);

    // 1D B-spline filter coefficients for each axis
    g0 = 0.5 - b;

    // 1D B-spline filter offsets for each axis
    vec3 h0 = (0.5 + b) * 0.5;

    // 1D B-spline filter normalized positions for each axis
    vec3 p = coords * t;
    p0 = p - h0 * t;
    p1 = p0 + 0.5 * t;
}

// Triquadratic B-spline interpolation samples

void triquadratic_bspline_samples(in sampler3D tex, in vec3 p0, in vec3 p1, out vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, out vec4 s_x1y0z0_x1y1z0_x1y0z1_x1y1z1)
{
    // Sample the 8 corner points of the interpolation cube based on p0, p1

    s_x0y0z0_x0y1z0_x0y0z1_x0y1z1 = vec4(
        texture(tex, vec3(p0.x, p0.y, p0.z)).r, // x0y0z0
        texture(tex, vec3(p0.x, p1.y, p0.z)).r, // x0y1z0
        texture(tex, vec3(p0.x, p0.y, p1.z)).r, // x0y0z1
        texture(tex, vec3(p0.x, p1.y, p1.z)).r  // x0y1z1
    );

    s_x1y0z0_x1y1z0_x1y0z1_x1y1z1 = vec4(
        texture(tex, vec3(p1.x, p0.y, p0.z)).r, // x1y0z0
        texture(tex, vec3(p1.x, p1.y, p0.z)).r, // x1y1z0
        texture(tex, vec3(p1.x, p0.y, p1.z)).r, // x1y0z1
        texture(tex, vec3(p1.x, p1.y, p1.z)).r  // x1y1z1
    );
}

// Triquadratic B-spline interpolation of value    

float triquadratic_bspline_xyz(in sampler3D tex, in vec3 p0, in vec3 p1, in vec3 g0)
{
    // Sample cube
    vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1;
    triquadratic_bspline_samples(tex, p0, p1, s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1);

    // Interpolate along x
    vec4 s_xy0z0_xy1z0_xy0z1_xy1z1 = mix(
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1, 
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, 
    g0.x);

    // Interpolate along y
    vec2 s_xyz0_xyz1 = mix(
        s_xy0z0_xy1z0_xy0z1_xy1z1.yw,
        s_xy0z0_xy1z0_xy0z1_xy1z1.xz,
    g0.y);

    // Interpolate along z
    float s_xyz = mix(
        s_xyz0_xyz1.y,
        s_xyz0_xyz1.x, 
    g0.z);

    // Intensity
    return s_xyz;
}

// Triquadratic B-spline interpolation of gradient

vec3 triquadratic_bspline_dxyz_xdyz_xydz(in sampler3D tex, in vec3 p0, in vec3 p1, in vec3 g0)
{
    // Sample cube
    vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1;
    triquadratic_bspline_samples(tex, p0, p1, s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1);

    // Interpolate along x
    vec4 s_xy0z0_xy1z0_xy0z1_xy1z1 = mix(
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1, 
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, 
    g0.x);

    // Differentiate across x
    vec4 s_dxy0z0_dxy1z0_dxy0z1_dxy1z1 = (
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1 - 
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1
    ) * 2.0;

    // Interpolate along y
    vec4 s_xyz0_xyz1_dxyz0_dxyz1 = mix(
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.yw, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.yw),
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.xz, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.xz),
    g0.y);

    // Differentiate across y
    vec2 s_xdyz0_xdyz1 = (
        s_xy0z0_xy1z0_xy0z1_xy1z1.yw - 
        s_xy0z0_xy1z0_xy0z1_xy1z1.xz
    ) * 2.0;

    // Interpolate along z
    vec2 s_dxyz_xdyz = mix(
        vec2(s_xyz0_xyz1_dxyz0_dxyz1.w, s_xdyz0_xdyz1.y),
        vec2(s_xyz0_xyz1_dxyz0_dxyz1.z, s_xdyz0_xdyz1.x), 
    g0.z);

    // Differentiate across z
    float s_xydz = (
        s_xyz0_xyz1_dxyz0_dxyz1.y - 
        s_xyz0_xyz1_dxyz0_dxyz1.x
    ) * 2.0;

    // Gradient
    return vec3(s_dxyz_xdyz, s_xydz);
}

// Triquadratic B-spline interpolation of hessian

vec3 triquadratic_bspline_xdydz_dxydz_dxdyz(in sampler3D tex, in vec3 p0, in vec3 p1, in vec3 g0)
{
    // Mixed second derivatives of xyz axes

    // Sample cube
    vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1;
    triquadratic_bspline_samples(tex, p0, p1, s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1);

    // Interpolate along x
    vec4 s_xy0z0_xy1z0_xy0z1_xy1z1 = mix(
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1, 
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, 
    g0.x);

    // Differentiate across x
    vec4 s_dxy0z0_dxy1z0_dxy0z1_dxy1z1 = (
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1 -
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1
    ) * 2.0;

    // Interpolate along y
    vec2 s_dxyz0_dxyz1 = mix(
        s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.yw,
        s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.xz,
    g0.y);

    // Differentiate across y
    vec4 s_xdyz0_xdyz1_dxdyz0_dxdyz1 = (
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.yw, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.yw) - 
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.xz, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.xz)
    ) * 2.0;

    // Interpolate along z
    float s_dxdyz = mix(
        s_xdyz0_xdyz1_dxdyz0_dxdyz1.w,
        s_xdyz0_xdyz1_dxdyz0_dxdyz1.z,
    g0.z);

    // Differentiate across z 
    vec2 s_xdydz_dxydz = (
        vec2(s_xdyz0_xdyz1_dxdyz0_dxdyz1.y, s_dxyz0_dxyz1.y) - 
        vec2(s_xdyz0_xdyz1_dxdyz0_dxdyz1.x, s_dxyz0_dxyz1.x)
    ) * 2.0;

    // Mixes derivatives
    return vec3(s_xdydz_dxydz, s_dxdyz);
}

vec3 triquadratic_bspline_d2x_d2y_d2z(in sampler3D tex, in vec3 coords)
{
    // Pure second derivatives of xyz axes

    // Get size, normalized position and step
    vec3 size = vec3(textureSize(tex, 0));
    vec3 t = 1.0 / size;
    vec3 p = coords * t;

    // Central differences samples
    float s = texture(tex, p).r;

    vec3 s_x0_y0_z0 = vec3(
        texture(tex, vec3(p.x - t.x, p.y, p.z)).r,
        texture(tex, vec3(p.x, p.y - t.y, p.z)).r,
        texture(tex, vec3(p.x, p.y, p.z - t.z)).r
    );

    vec3 s_x1_y1_z1 = vec3(
        texture(tex, vec3(p.x + t.x, p.y, p.z)).r,
        texture(tex, vec3(p.x, p.y + t.y, p.z)).r,
        texture(tex, vec3(p.x, p.y, p.z + t.z)).r
    );

    // Pure second derivatives
    vec3 s_d2x_d2y_d2z = s_x0_y0_z0 + s_x1_y1_z1 - s * 2.0;

    return s_d2x_d2y_d2z;
}

// Triquadratic B-spline interpolations

void triquadratic_bspline_value(in sampler3D tex, in vec3 coords, out float value)
{
    vec3 p0, p1, g0;
    triquadratic_bspline_basis(tex, coords, p0, p1, g0);

    // Value
    value = triquadratic_bspline_xyz(tex, p0, p1, g0);
}

void triquadratic_bspline_gradient(in sampler3D tex, in vec3 coords,  out vec3 gradient)
{
    vec3 p0, p1, g0;
    triquadratic_bspline_basis(tex, coords, p0, p1, g0);
 
    // Gradient
    gradient = triquadratic_bspline_dxyz_xdyz_xydz(tex, p0, p1, g0);
}
 
void triquadratic_bspline_hessian(in sampler3D tex, in vec3 coords, out mat3 hessian)
{
    vec3 p0, p1, g0; 
    triquadratic_bspline_basis(tex, coords, p0, p1, g0);
 
    // Mixed derivatives
    vec3 s_xdydz_dxydz_dxdyz = triquadratic_bspline_xdydz_dxydz_dxdyz(tex, p0, p1, g0);

    // Pure derivatives
    vec3 s_d2x_d2y_d2z = triquadratic_bspline_d2x_d2y_d2z(tex, coords);

    // Hessian
    hessian = mat3(
       s_d2x_d2y_d2z.x, s_xdydz_dxydz_dxdyz.z, s_xdydz_dxydz_dxdyz.y,  
       s_xdydz_dxydz_dxdyz.z, s_d2x_d2y_d2z.y, s_xdydz_dxydz_dxdyz.x,  
       s_xdydz_dxydz_dxdyz.y, s_xdydz_dxydz_dxdyz.x, s_d2x_d2y_d2z.z     
   );
}

void triquadratic_bspline_value_hessian(in sampler3D tex, in vec3 coords, out float value, out mat3 hessian)
{
    vec3 p0, p1, g0;
    triquadratic_bspline_basis(tex, coords, p0, p1, g0);

    // Sample cube
    vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1;
    triquadratic_bspline_samples(tex, p0, p1, s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1);

    // Interpolate along x
    vec4 s_xy0z0_xy1z0_xy0z1_xy1z1 = mix(
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1, 
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, 
    g0.x);

    // Differentiate across x
    vec4 s_dxy0z0_dxy1z0_dxy0z1_dxy1z1 = (
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1 - 
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1
    ) * 2.0;

    // Interpolate along y
    vec4 s_xyz0_xyz1_dxyz0_dxyz1 = mix(
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.yw, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.yw),
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.xz, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.xz),
    g0.y);

    // Differentiate across y
    vec4 s_xdyz0_xdyz1_dxdyz0_dxdyz1 = (
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.yw, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.yw) - 
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.xz, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.xz)
    ) * 2.0;

    // Interpolate along z
    vec2 s_xyz_dxdyz = mix(
        vec2(s_xyz0_xyz1_dxyz0_dxyz1.y, s_xdyz0_xdyz1_dxdyz0_dxdyz1.w),
        vec2(s_xyz0_xyz1_dxyz0_dxyz1.x, s_xdyz0_xdyz1_dxdyz0_dxdyz1.z),
    g0.z);

    // Differentiate across z
    vec2 s_dxydz_xdydz = (
        vec2(s_xyz0_xyz1_dxyz0_dxyz1.w, s_xdyz0_xdyz1_dxdyz0_dxdyz1.y) -
        vec2(s_xyz0_xyz1_dxyz0_dxyz1.z, s_xdyz0_xdyz1_dxdyz0_dxdyz1.x)
    ) * 2.0;

    // Pure derivatives
    vec3 s_d2x_d2y_d2z = triquadratic_bspline_d2x_d2y_d2z(tex, coords);

    // Value
    value = s_xyz_dxdyz.x;

    // Hessian
    hessian = mat3(
       s_d2x_d2y_d2z.x, s_xyz_dxdyz.y, s_dxydz_xdydz.x,  
       s_xyz_dxdyz.y, s_d2x_d2y_d2z.y, s_dxydz_xdydz.y,  
       s_dxydz_xdydz.x, s_dxydz_xdydz.y, s_d2x_d2y_d2z.z     
   );
}

void triquadratic_bspline_value_gradient(in sampler3D tex, in vec3 coords, out float value, out vec3 gradient)
{
    vec3 p0, p1, g0;
    triquadratic_bspline_basis(tex, coords, p0, p1, g0);

    // Sample cube
    vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1;
    triquadratic_bspline_samples(tex, p0, p1, s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1);

    // Interpolate along x
    vec4 s_xy0z0_xy1z0_xy0z1_xy1z1 = mix(
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1, 
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, 
    g0.x);

    // Differentiate across x
    vec4 s_dxy0z0_dxy1z0_dxy0z1_dxy1z1 = (
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1 - 
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1
    ) * 2.0;

    // Interpolate along y
    vec4 s_xyz0_xyz1_dxyz0_dxyz1 = mix(
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.yw, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.yw),
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.xz, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.xz),
    g0.y);

    // Differentiate across y
    vec2 s_xdyz0_xdyz1 = (
        s_xy0z0_xy1z0_xy0z1_xy1z1.yw - 
        s_xy0z0_xy1z0_xy0z1_xy1z1.xz
    ) * 2.0;

    // Interpolate along z
    vec3 s_xyz_dxyz_xdyz = mix(
        vec3(s_xyz0_xyz1_dxyz0_dxyz1.yw, s_xdyz0_xdyz1.y),
        vec3(s_xyz0_xyz1_dxyz0_dxyz1.xz, s_xdyz0_xdyz1.x), 
    g0.z);

    // Differentiate across z
    float s_xydz = (
        s_xyz0_xyz1_dxyz0_dxyz1.y - 
        s_xyz0_xyz1_dxyz0_dxyz1.x
    ) * 2.0;

    // Value
    value = s_xyz_dxyz_xdyz.x;

    // Gradient
    gradient = vec3(s_xyz_dxyz_xdyz.yz, s_xydz);
}

void triquadratic_bspline_gradient_hessian(in sampler3D tex, in vec3 coords, out vec3 gradient, out mat3 hessian)
{
    vec3 p0, p1, g0;
    triquadratic_bspline_basis(tex, coords, p0, p1, g0);

    // Sample cube
    vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1;
    triquadratic_bspline_samples(tex, p0, p1, s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1);

    // Interpolate along x
    vec4 s_xy0z0_xy1z0_xy0z1_xy1z1 = mix(
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1, 
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, 
    g0.x);

    // Differentiate across x
    vec4 s_dxy0z0_dxy1z0_dxy0z1_dxy1z1 = (
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1 - 
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1
    ) * 2.0;

    // Interpolate along y
    vec4 s_xyz0_xyz1_dxyz0_dxyz1 = mix(
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.yw, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.yw),
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.xz, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.xz),
    g0.y);

    // Differentiate across y
    vec4 s_xdyz0_xdyz1_dxdyz0_dxdyz1 = (
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.yw, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.yw) -
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.xz, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.xz)
    ) * 2.0;

    // Interpolate along z
    vec3 s_dxyz_xdyz_dxdyz = mix(
        vec3(s_xyz0_xyz1_dxyz0_dxyz1.w, s_xdyz0_xdyz1_dxdyz0_dxdyz1.yw),
        vec3(s_xyz0_xyz1_dxyz0_dxyz1.z, s_xdyz0_xdyz1_dxdyz0_dxdyz1.xz), 
    g0.z);

    // Differentiate across z
    vec3 s_xydz_dxydz_xdydz = (
        vec3(s_xyz0_xyz1_dxyz0_dxyz1.yw, s_xdyz0_xdyz1_dxdyz0_dxdyz1.y) -
        vec3(s_xyz0_xyz1_dxyz0_dxyz1.xz, s_xdyz0_xdyz1_dxdyz0_dxdyz1.x)
    ) * 2.0;

    // Pure derivatives
    vec3 s_d2x_d2y_d2z = triquadratic_bspline_d2x_d2y_d2z(tex, coords);

    // Gradient
    gradient = vec3(s_dxyz_xdyz_dxdyz.xy, s_xydz_dxydz_xdydz.x);

    // Hessian
    hessian = mat3(
       s_d2x_d2y_d2z.x, s_dxyz_xdyz_dxdyz.z, s_xydz_dxydz_xdydz.y,  
       s_dxyz_xdyz_dxdyz.z, s_d2x_d2y_d2z.y, s_xydz_dxydz_xdydz.z,  
       s_xydz_dxydz_xdydz.y, s_xydz_dxydz_xdydz.z, s_d2x_d2y_d2z.z     
   );
}

void triquadratic_bspline_value_gradient_hessian(in sampler3D tex, in vec3 coords, out float value, out vec3 gradient, out mat3 hessian)
{
    vec3 p0, p1, g0;
    triquadratic_bspline_basis(tex, coords, p0, p1, g0);

    // Sample cube
    vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1;
    triquadratic_bspline_samples(tex, p0, p1, s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1);

    // Interpolate along x
    vec4 s_xy0z0_xy1z0_xy0z1_xy1z1 = mix(
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1, 
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, 
    g0.x);

    // Differentiate across x
    vec4 s_dxy0z0_dxy1z0_dxy0z1_dxy1z1 = (
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1 - 
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1
    ) * 2.0;

    // Interpolate along y
    vec4 s_xyz0_xyz1_dxyz0_dxyz1 = mix(
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.yw, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.yw),
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.xz, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.xz),
    g0.y);

    // Differentiate across y
    vec4 s_xdyz0_xdyz1_dxdyz0_dxdyz1 = (
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.yw, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.yw) - 
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.xz, s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.xz)
    ) * 2.0;

    // Interpolate along z
    vec4 s_xyz_dxyz_xdyz_dxdyz = mix(
        vec4(s_xyz0_xyz1_dxyz0_dxyz1.yw, s_xdyz0_xdyz1_dxdyz0_dxdyz1.yw),
        vec4(s_xyz0_xyz1_dxyz0_dxyz1.xz, s_xdyz0_xdyz1_dxdyz0_dxdyz1.xz),
    g0.z);

    // Differentiate across z
    vec3 s_xydz_dxydz_xdydz = (
        vec3(s_xyz0_xyz1_dxyz0_dxyz1.yw, s_xdyz0_xdyz1_dxdyz0_dxdyz1.y) -
        vec3(s_xyz0_xyz1_dxyz0_dxyz1.xz, s_xdyz0_xdyz1_dxdyz0_dxdyz1.x)
    ) * 2.0;

    // Pure derivatives
    vec3 s_d2x_d2y_d2z = triquadratic_bspline_d2x_d2y_d2z(tex, coords);

    // Value
    value = s_xyz_dxyz_xdyz_dxdyz.x;

    // Gradient
    gradient = vec3(s_xyz_dxyz_xdyz_dxdyz.yz, s_xydz_dxydz_xdydz.x);

    // Hessian
    hessian = mat3(
       s_d2x_d2y_d2z.x, s_xyz_dxyz_xdyz_dxdyz.w, s_xydz_dxydz_xdydz.y,  
       s_xyz_dxyz_xdyz_dxdyz.w, s_d2x_d2y_d2z.y, s_xydz_dxydz_xdydz.z,  
       s_xydz_dxydz_xdydz.y, s_xydz_dxydz_xdydz.z, s_d2x_d2y_d2z.z     
   );
}

#endif