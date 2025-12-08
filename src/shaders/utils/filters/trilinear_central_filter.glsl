#ifndef TRILINEAR_SOBEL_FILTER
#define TRILINEAR_SOBEL_FILTER

/* Sources
MRIcroGL Gradients
(https://github.com/neurolabusc/blog/blob/main/GL-gradients/README.md),

GPU Gems 2, Chapter 20. Fast Third-Order Texture Filtering 
(https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering),
*/

// Trilinear central interpolation samples

void trilinear_central_samples(in sampler3D tex, in vec3 coords, out vec3 s_x0y1z1_x1y0z1_x1y1z0, out vec3 s_x2y1z1_x1y2z1_x1y1z2)
{
    // Sample the 6 central differences

    // Get size, normalized position and step
    vec3 size = vec3(textureSize(tex, 0));
    vec3 t = 1.0 / size;
    vec3 p = coords * t;

    // Central differences samples
    s_x0y1z1_x1y0z1_x1y1z0 = vec3(
        texture(tex, vec3(p.x - t.x, p.y, p.z)).r, //x0y1z1
        texture(tex, vec3(p.x, p.y - t.y, p.z)).r, //x1y0z1
        texture(tex, vec3(p.x, p.y, p.z - t.z)).r  //x1y1z0
    );

    s_x2y1z1_x1y2z1_x1y1z2 = vec3(
        texture(tex, vec3(p.x + t.x, p.y, p.z)).r, //x2y1z1
        texture(tex, vec3(p.x, p.y + t.y, p.z)).r, //x1y2z1
        texture(tex, vec3(p.x, p.y, p.z + t.z)).r  //x1y1z2
    );
}

void trilinear_central_samples(in sampler3D tex, in vec3 coords, out float s_x1y1z1, out vec3 s_x0y1z1_x1y0z1_x1y1z0, out vec3 s_x2y1z1_x1y2z1_x1y1z2)
{
    // Sample the 6 central differences and center

    // Get size, normalized position and step
    vec3 size = vec3(textureSize(tex, 0));
    vec3 t = 1.0 / size;
    vec3 p = coords * t;

    // Center sample
    s_x1y1z1 = texture(tex, p).r; // x1y1z1

    // Central differences samples
    s_x0y1z1_x1y0z1_x1y1z0 = vec3(
        texture(tex, vec3(p.x - t.x, p.y, p.z)).r, //x0y1z1
        texture(tex, vec3(p.x, p.y - t.y, p.z)).r, //x1y0z1
        texture(tex, vec3(p.x, p.y, p.z - t.z)).r  //x1y1z0
    );

    s_x2y1z1_x1y2z1_x1y1z2 = vec3(
        texture(tex, vec3(p.x + t.x, p.y, p.z)).r, //x2y1z1
        texture(tex, vec3(p.x, p.y + t.y, p.z)).r, //x1y2z1
        texture(tex, vec3(p.x, p.y, p.z + t.z)).r  //x1y1z2
    );
}

// Trilinear central interpolations

void trilinear_central_gradient(in sampler3D tex, in vec3 coords, out vec3 gradient)
{
    // Sample central cross
    vec3 s_x0y1z1_x1y0z1_x1y1z0, s_x2y1z1_x1y2z1_x1y1z2;
    trilinear_central_samples(tex, coords, s_x0y1z1_x1y0z1_x1y1z0, s_x2y1z1_x1y2z1_x1y1z2);

    // First order central differences across xyz
    vec3 s_dx_dy_dz = (s_x2y1z1_x1y2z1_x1y1z2 - s_x0y1z1_x1y0z1_x1y1z0) * 0.5;

    // Gradient
    gradient = s_dx_dy_dz;
}

void trilinear_central_laplacian(in sampler3D tex, in vec3 coords, out vec3 laplacian)
{
    // Pure second derivatives of xyz axes

    // Sample central cross
    float s_x1y1z1; vec3 s_x0y1z1_x1y0z1_x1y1z0, s_x2y1z1_x1y2z1_x1y1z2;
    trilinear_central_samples(tex, coords, s_x1y1z1, s_x0y1z1_x1y0z1_x1y1z0, s_x2y1z1_x1y2z1_x1y1z2);

    // Second order finite differences across xyz
    vec3 s_d2x_d2y_d2z = s_x2y1z1_x1y2z1_x1y1z2 + s_x0y1z1_x1y0z1_x1y1z0 - s_x1y1z1 * 2.0;

    // Laplacian
    laplacian = s_d2x_d2y_d2z;
}

void trilinear_central_gradient_laplacian(in sampler3D tex, in vec3 coords, out vec3 gradient, out vec3 laplacian)
{
    // Pure second derivatives of xyz axes

    // Sample central cross
    float s_x1y1z1; vec3 s_x0y1z1_x1y0z1_x1y1z0, s_x2y1z1_x1y2z1_x1y1z2;
    trilinear_central_samples(tex, coords, s_x1y1z1, s_x0y1z1_x1y0z1_x1y1z0, s_x2y1z1_x1y2z1_x1y1z2);

    // First order central differences across xyz
    gradient = (s_x2y1z1_x1y2z1_x1y1z2 - s_x0y1z1_x1y0z1_x1y1z0) * 0.5;

    // Second order central differences across xyz
    laplacian = s_x2y1z1_x1y2z1_x1y1z2 + s_x0y1z1_x1y0z1_x1y1z0 - s_x1y1z1 * 2.0;
}

// Triquadratic central interpolation from paper
// Beyond Trilinear Interpolation: Higher Quality for Free
// (https://dl.acm.org/doi/10.1145/3306346.3323032),

void triquadratic_central_value(in sampler3D tex, in vec3 coords, out float value)
{
    // Sample central cross
    float s_x1y1z1; vec3 s_x0y1z1_x1y0z1_x1y1z0, s_x2y1z1_x1y2z1_x1y1z2;
    trilinear_central_samples(tex, coords, s_x1y1z1, s_x0y1z1_x1y0z1_x1y1z0, s_x2y1z1_x1y2z1_x1y1z2);

    // Second order finite differences across xyz
    vec3 s_d2x_d2y_d2z = s_x2y1z1_x1y2z1_x1y1z2 + s_x0y1z1_x1y0z1_x1y1z0 - s_x1y1z1 * 2.0;

    // Convert to voxel-space and compute local coordinates
    vec3 x = coords - 0.5;
    vec3 a = x - floor(x);
    vec3 g = (a * (a - 1.0)) / 2.0;

    // Compute triquadratic correction
    float s_xyz = s_x1y1z1 + dot(s_d2x_d2y_d2z, g);

    // Value
    value = s_xyz;
}

void triquadratic_central_value_gradient(in sampler3D tex, in vec3 coords, out float value, out vec3 gradient)
{
    // Sample central cross
    float s_x1y1z1; vec3 s_x0y1z1_x1y0z1_x1y1z0, s_x2y1z1_x1y2z1_x1y1z2;
    trilinear_central_samples(tex, coords, s_x1y1z1, s_x0y1z1_x1y0z1_x1y1z0, s_x2y1z1_x1y2z1_x1y1z2);

    // First order central differences across xyz
    vec3 s_dx_dy_dz = (s_x2y1z1_x1y2z1_x1y1z2 - s_x0y1z1_x1y0z1_x1y1z0) * 0.5;

    // Second order finite differences across xyz
    vec3 s_d2x_d2y_d2z = s_x2y1z1_x1y2z1_x1y1z2 + s_x0y1z1_x1y0z1_x1y1z0 - s_x1y1z1 * 2.0;

    // Convert to voxel-space and compute local coordinates
    vec3 x = coords - 0.5;
    vec3 a = x - floor(x);
    vec3 g = (a * (a - 1.0)) / 2.0;

    // Compute triquadratic correction
    float s_xyz = s_x1y1z1 + dot(s_d2x_d2y_d2z, g);

    // Value
    value = s_xyz;

    // Gradient
    gradient = s_dx_dy_dz;
}

#endif