#ifndef TRICUBIC_BSPLINE_FILTER
#define TRICUBIC_BSPLINE_FILTER

/* Sources
Efficient GPU-Based Texture Interpolation using Uniofm B-Splines  
(https://www.tandfonline.com/doi/abs/10.1080/2151237X.2008.10129269),

GPU Gems 2, Chapter 20. Fast Third-Order Texture Filtering 
(https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering),
*/

// Tricubic B-spline interpolation basis

void tricubic_bspline_basis(in sampler3D tex, in vec3 coords, out vec3 p0, out vec3 p1, out vec3 g0)
{
    // Get size and normalized step
    vec3 size = vec3(textureSize(tex, 0));
    vec3 t = 1.0 / size;

    // Convert to voxel-space and compute local coordinates
    vec3 x = coords - 0.5;
    vec3 i = floor(x);
    vec3 a = x - i;

    // Pre-computations
    vec3 one_a = 1.0 - a;
    vec3 one_a2 = one_a * one_a;
    vec3 a2 = a * a;

    // 1D B-spline filter weights for each axis
    vec3 w0 = (1.0/6.0) * one_a2 * one_a;
    vec3 w1 = (2.0/3.0) - 0.5 * a2 * (2.0 - a);
    vec3 w2 = (2.0/3.0) - 0.5 * one_a2 * (2.0 - one_a);
    vec3 w3 = (1.0/6.0) * a2 * a;

    // 1D B-spline filter coefficients for each axis
    g0 = w0 + w1;
    vec3 g1 = w2 + w3;

    // 1D B-spline filter position for each axis
    vec3 h0 = (w1 / g0) - 0.5 + i; // h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
    vec3 h1 = (w3 / g1) + 1.5 + i;

    // 1D B-spline filter normalized position for each axis
    p0 = h0 * t;
    p1 = h1 * t;
}

void tricubic_bspline_basis(in sampler3D tex, in vec3 coords, out vec3 p0, out vec3 p1, out vec3 g0, out vec3 dp0, out vec3 dp1, out vec3 dg0)
{
    // Get size and normalized step
    vec3 size = vec3(textureSize(tex, 0));
    vec3 t = 1.0 / size;

    // Convert to voxel-space and compute local coordinates
    vec3 x = coords - 0.5;
    vec3 i = floor(x);
    vec3 a = x - i;

    // Pre-computations
    vec3 one_a = 1.0 - a;
    vec3 one_a2 = one_a * one_a;
    vec3 a2 = a * a;

    // 1D B-spline weights of each axis
    vec3 w0 = (1.0/6.0) * one_a2 * one_a;
    vec3 w1 = (2.0/3.0) - 0.5 * a2 * (2.0 - a);
    vec3 w2 = (2.0/3.0) - 0.5 * one_a2 * (2.0 - one_a);
    vec3 w3 = (1.0/6.0) * a2 * a;

    // 1D derivative B-spline weights of each axis
    vec3 dw0 = -0.5 * one_a2;
    vec3 dw1 = -0.5 * a * (3.0 * one_a + 1.0);
    vec3 dw2 = 0.5 * one_a * (3.0 * a + 1.0);
    vec3 dw3 = 0.5 * a2;

    // 1D B-spline filter coefficients for each axis
    g0 = w0 + w1;
    vec3 g1 = w2 + w3;
    
    // 1D B-spline derivative filter coefficients for each axis
    dg0 = dw0 + dw1;
    vec3 dg1 = dw2 + dw3;

    // 1D B-spline filter position for each axis
    vec3 h0 = (w1 / g0) - 0.5 + i; // h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
    vec3 h1 = (w3 / g1) + 1.5 + i;

    // 1D B-spline derivative filter position for each axis
    vec3 dh0 = (dw1 / dg0) - 0.5 + i;
    vec3 dh1 = (dw3 / dg1) + 1.5 + i;

    // 1D B-spline filter normalized position for each axis
    p0 = h0 * t;
    p1 = h1 * t;

    // 1D B-spline derivative filter normalized position for each axis
    dp0 = dh0 * t;
    dp1 = dh1 * t;
}

// Tricubic B-spline interpolation samples

void tricubic_bspline_samples(in sampler3D tex, in vec3 p0, in vec3 p1, out vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, out vec4 s_x1y0z0_x1y1z0_x1y0z1_x1y1z1)
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

// Tricubic B-spline interpolation of value    

float tricubic_bspline_xyz(in sampler3D tex, in vec3 p0, in vec3 p1, in vec3 g0)
{
    // Sample cube
    vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1;
    tricubic_bspline_samples(tex, p0, p1, s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1);

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

    return s_xyz;
}

// Tricubic B-spline interpolation of first derivatives

float tricubic_bspline_dxyz(in sampler3D tex, in vec3 p0, in vec3 p1, in vec3 g0)
{
    // First derivative of x axis

    // Sample cube
    vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1;
    tricubic_bspline_samples(tex, p0, p1, s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1);

    // Differentiate along x
    vec4 s_dxy0z0_dxy1z0_dxy0z1_dxy1z1 = (
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1 -
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1 
    ) * g0.x;

    // Interpolate along y
    vec2 s_dxyz0_dxyz1 = mix(
        s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.yw,
        s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.xz,
    g0.y);

    // Interpolate along z
    float s_dxyz = mix(
        s_dxyz0_dxyz1.y,
        s_dxyz0_dxyz1.x, 
    g0.z);

    return s_dxyz;
}

float tricubic_bspline_xdyz(in sampler3D tex, in vec3 p0, in vec3 p1, in vec3 g0)
{
    // First derivative of y axis

    // Sample cube
    vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1;
    tricubic_bspline_samples(tex, p0, p1, s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1);

    // Interpolate along x
    vec4 s_xy0z0_xy1z0_xy0z1_xy1z1 = mix(
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1,
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1,
    g0.x);

    // Differentiate along y
    vec2 s_xdyz0_xdyz1 = (
        s_xy0z0_xy1z0_xy0z1_xy1z1.xz -
        s_xy0z0_xy1z0_xy0z1_xy1z1.yw
    ) * g0.y;

    // Interpolate along z
    float s_xdyz = mix(
        s_xdyz0_xdyz1.y,
        s_xdyz0_xdyz1.x, 
    g0.z);

    return s_xdyz;
}

float tricubic_bspline_xydz(in sampler3D tex, in vec3 p0, in vec3 p1, in vec3 g0)
{
    // First derivative of z axis
    
    // Sample cube
    vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1;
    tricubic_bspline_samples(tex, p0, p1, s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1);

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

    // Differentiate along z
    float s_xydz = (
        s_xyz0_xyz1.x -
        s_xyz0_xyz1.y
    ) * g0.z;

    return s_xydz;
}

vec3 tricubic_bspline_dxyz_xdyz_xydz(in sampler3D tex, in vec3 p0, in vec3 p1, in vec3 g0, in vec3 dp0, in vec3 dp1, in vec3 dg0)
{
    // First derivative of x axis
    float s_dxyz = tricubic_bspline_dxyz(tex, 
        vec3(dp0.x, p0.y, p0.z), 
        vec3(dp1.x, p1.y, p1.z), 
        vec3(dg0.x, g0.y, g0.z)
    );

    // First derivative of x axis
    float s_xdyz = tricubic_bspline_xdyz(tex, 
        vec3(p0.x, dp0.y, p0.z), 
        vec3(p1.x, dp1.y, p1.z), 
        vec3(g0.x, dg0.y, g0.z)
    );

    // First derivative of x axis
    float s_xydz = tricubic_bspline_xydz(tex, 
        vec3(p0.x, p0.y, dp0.z), 
        vec3(p1.x, p1.y, dp1.z), 
        vec3(g0.x, g0.y, dg0.z)
    );

    // Gradient
    return vec3(s_dxyz, s_xdyz, s_xydz);
}

// Tricubic B-spline interpolation of second derivatives

float tricubic_bspline_xdydz(in sampler3D tex, in vec3 p0, in vec3 p1, in vec3 g0)
{
    // Mixed second derivative of yz axes

    // Sample cube
    vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1;
    tricubic_bspline_samples(tex, p0, p1, s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1);

    // Interpolate along x
    vec4 s_xy0z0_xy1z0_xy0z1_xy1z1 = mix(
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1,
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1,
    g0.x);

    // Differentiate along y
    vec2 s_xdyz0_xdyz1 = (
        s_xy0z0_xy1z0_xy0z1_xy1z1.xz -
        s_xy0z0_xy1z0_xy0z1_xy1z1.yw
    ) * g0.y;

    // Differentiate along z
    float s_xdydz = (
        s_xdyz0_xdyz1.x -
        s_xdyz0_xdyz1.y
    ) *  g0.z;

    return s_xdydz;
}

float tricubic_bspline_dxydz(in sampler3D tex, in vec3 p0, in vec3 p1, in vec3 g0)
{
    // Mixed second derivative of xz axes

    // Sample cube
    vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1;
    tricubic_bspline_samples(tex, p0, p1, s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1);

    // Differentiate along x
    vec4 s_dxy0z0_dxy1z0_dxy0z1_dxy1z1 = (
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1 -
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1
    ) * g0.x;

    // Interpolate along y
    vec2 s_dxyz0_dxyz1 = mix(
        s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.yw,
        s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.xz,
    g0.y);

    // Differentiate along z
    float s_dxydz = (
        s_dxyz0_dxyz1.x -
        s_dxyz0_dxyz1.y
    ) * g0.z;

    return s_dxydz;
}

float tricubic_bspline_dxdyz(in sampler3D tex, in vec3 p0, in vec3 p1, in vec3 g0)
{
    // Mixed second derivative of xy axes

    // Sample cube
    vec4 s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1;
    tricubic_bspline_samples(tex, p0, p1, s_x0y0z0_x0y1z0_x0y0z1_x0y1z1, s_x1y0z0_x1y1z0_x1y0z1_x1y1z1);

     // Differentiate along x
    vec4 s_dxy0z0_dxy1z0_dxy0z1_dxy1z1 = (
        s_x0y0z0_x0y1z0_x0y0z1_x0y1z1 -
        s_x1y0z0_x1y1z0_x1y0z1_x1y1z1
    ) * g0.x;

    // Differentiate along y
    vec2 s_dxdyz0_dxdyz1 = (
        s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.xz -
        s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.yw
    ) * g0.y;

    // Interpolate along z
    float s_dxdyz = mix(
        s_dxdyz0_dxdyz1.y, 
        s_dxdyz0_dxdyz1.x, 
    g0.z);
    
    return s_dxdyz;
}

vec3 tricubic_bspline_xdydz_dxydz_dxdyz(in sampler3D tex, in vec3 p0, in vec3 p1, in vec3 g0, in vec3 dp0, in vec3 dp1, in vec3 dg0)
{
    // Mixed second derivative of yz axes
    float s_xdydz = tricubic_bspline_xdydz(tex, 
        vec3(p0.x, dp0.y, dp0.z), 
        vec3(p1.x, dp1.y, dp1.z), 
        vec3(g0.x, dg0.y, dg0.z)
    );

    // Mixed second derivative of xz axes
    float s_dxydz = tricubic_bspline_dxydz(tex, 
        vec3(dp0.x, p0.y, dp0.z), 
        vec3(dp1.x, p1.y, dp1.z), 
        vec3(dg0.x, g0.y, dg0.z)
    );

    // Mixed second derivative of xy axes
    float s_dxdyz = tricubic_bspline_dxdyz(tex, 
        vec3(dp0.x, dp0.y, p0.z), 
        vec3(dp1.x, dp1.y, p1.z), 
        vec3(dg0.x, dg0.y, g0.z)
    );

    // Mixed second derivatives
    return vec3(s_xdydz, s_dxydz, s_dxdyz);
}

vec3 tricubic_bspline_d2x_d2y_d2z(in sampler3D tex,  in vec3 coords)
{
    // Pure second derivatives of xyz aces

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
    return s_x0_y0_z0 + s_x1_y1_z1 - s * 2.0;
}

// Tricubic B-spline combined interpolations

void tricubic_bspline_value(in sampler3D tex, in vec3 coords, out float value)
{
    vec3 p0; vec3 p1; vec3 g0;
    tricubic_bspline_basis(tex, coords, p0, p1, g0);

    // Value
    value = tricubic_bspline_xyz(tex, p0, p1, g0);
}

void tricubic_bspline_gradient(in sampler3D tex, in vec3 coords, out vec3 gradient)
{
    vec3 p0; vec3 p1; vec3 g0; vec3 dp0; vec3 dp1; vec3 dg0;
    tricubic_bspline_basis(tex, coords, p0, p1, g0, dp0, dp1, dg0);
 
    // Gradient
    gradient = tricubic_bspline_dxyz_xdyz_xydz(tex, p0, p1, g0, dp0, dp1, dg0);
}

void tricubic_bspline_hessian(in sampler3D tex, in vec3 coords, out mat3 hessian)
{
    vec3 p0; vec3 p1; vec3 g0; vec3 dp0; vec3 dp1; vec3 dg0;
    tricubic_bspline_basis(tex, coords, p0, p1, g0, dp0, dp1, dg0);
 
    // Mixed second derivatives
    vec3 s_xdydz_dxydz_dxdyz = tricubic_bspline_xdydz_dxydz_dxdyz(tex, p0, p1, g0, dp0, dp1, dg0);
 
    // Pure second derivatives
    vec3 s_d2x_d2y_d2z = tricubic_bspline_d2x_d2y_d2z(tex, coords);

    // Hessian
    hessian = mat3(
       s_d2x_d2y_d2z.x, s_xdydz_dxydz_dxdyz.z, s_xdydz_dxydz_dxdyz.y,  
       s_xdydz_dxydz_dxdyz.z, s_d2x_d2y_d2z.y, s_xdydz_dxydz_dxdyz.x,  
       s_xdydz_dxydz_dxdyz.y, s_xdydz_dxydz_dxdyz.x, s_d2x_d2y_d2z.z     
   );
}

void tricubic_bspline_value_gradient(in sampler3D tex, in vec3 coords, out float value, out vec3 gradient)
{
    vec3 p0; vec3 p1; vec3 g0; vec3 dp0; vec3 dp1; vec3 dg0;
    tricubic_bspline_basis(tex, coords, p0, p1, g0, dp0, dp1, dg0);
 
    // Value
    value = tricubic_bspline_xyz(tex, p0, p1, g0);

    // Gradient
    gradient = tricubic_bspline_dxyz_xdyz_xydz(tex, p0, p1, g0, dp0, dp1, dg0);
}

void tricubic_bspline_value_hessian(in sampler3D tex, in vec3 coords, out float value, out mat3 hessian)
{
    vec3 p0; vec3 p1; vec3 g0; vec3 dp0; vec3 dp1; vec3 dg0;
    tricubic_bspline_basis(tex, coords, p0, p1, g0, dp0, dp1, dg0);
 
    // Value
    value = tricubic_bspline_xyz(tex, p0, p1, g0);

    // Mixed second derivatives
    vec3 s_xdydz_dxydz_dxdyz = tricubic_bspline_xdydz_dxydz_dxdyz(tex, p0, p1, g0, dp0, dp1, dg0);
 
    // Pure second derivatives
    vec3 s_d2x_d2y_d2z = tricubic_bspline_d2x_d2y_d2z(tex, coords);

    // Hessian
    hessian = mat3(
       s_d2x_d2y_d2z.x, s_xdydz_dxydz_dxdyz.z, s_xdydz_dxydz_dxdyz.y,  
       s_xdydz_dxydz_dxdyz.z, s_d2x_d2y_d2z.y, s_xdydz_dxydz_dxdyz.x,  
       s_xdydz_dxydz_dxdyz.y, s_xdydz_dxydz_dxdyz.x, s_d2x_d2y_d2z.z     
   );
}

void tricubic_bspline_gradient_hessian(in sampler3D tex, in vec3 coords, out vec3 gradient, out mat3 hessian)
{
    vec3 p0; vec3 p1; vec3 g0; vec3 dp0; vec3 dp1; vec3 dg0;
    tricubic_bspline_basis(tex, coords, p0, p1, g0, dp0, dp1, dg0);
 
    // Gradient
    gradient = tricubic_bspline_dxyz_xdyz_xydz(tex, p0, p1, g0, dp0, dp1, dg0); 

    // Mixed second derivatives
    vec3 s_xdydz_dxydz_dxdyz = tricubic_bspline_xdydz_dxydz_dxdyz(tex, p0, p1, g0, dp0, dp1, dg0);
 
    // Pure second derivatives
    vec3 s_d2x_d2y_d2z = tricubic_bspline_d2x_d2y_d2z(tex, coords);

    // Hessian
    hessian = mat3(
       s_d2x_d2y_d2z.x, s_xdydz_dxydz_dxdyz.z, s_xdydz_dxydz_dxdyz.y,  
       s_xdydz_dxydz_dxdyz.z, s_d2x_d2y_d2z.y, s_xdydz_dxydz_dxdyz.x,  
       s_xdydz_dxydz_dxdyz.y, s_xdydz_dxydz_dxdyz.x, s_d2x_d2y_d2z.z     
   );
}

void tricubic_bspline_value_gradient_hessian(in sampler3D tex, in vec3 coords, out float value, out vec3 gradient, out mat3 hessian)
{
    vec3 p0; vec3 p1; vec3 g0; vec3 dp0; vec3 dp1; vec3 dg0;
    tricubic_bspline_basis(tex, coords, p0, p1, g0, dp0, dp1, dg0);

    // Value
    value = tricubic_bspline_xyz(tex, p0, p1, g0);

    // Gradient
    gradient = tricubic_bspline_dxyz_xdyz_xydz(tex, p0, p1, g0, dp0, dp1, dg0); 

    // Mixed second derivatives
    vec3 s_xdydz_dxydz_dxdyz = tricubic_bspline_xdydz_dxydz_dxdyz(tex, p0, p1, g0, dp0, dp1, dg0);
 
    // Pure second derivatives
    vec3 s_d2x_d2y_d2z = tricubic_bspline_d2x_d2y_d2z(tex, coords);

    // Hessian
    hessian = mat3(
       s_d2x_d2y_d2z.x, s_xdydz_dxydz_dxdyz.z, s_xdydz_dxydz_dxdyz.y,  
       s_xdydz_dxydz_dxdyz.z, s_d2x_d2y_d2z.y, s_xdydz_dxydz_dxdyz.x,  
       s_xdydz_dxydz_dxdyz.y, s_xdydz_dxydz_dxdyz.x, s_d2x_d2y_d2z.z     
   );
}

#endif

