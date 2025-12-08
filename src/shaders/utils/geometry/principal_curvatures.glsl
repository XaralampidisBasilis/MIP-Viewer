// From the paper "Real-Time Ray-Casting and Advanced Shading of Discrete Isosurfaces"
// in the 4.1. Differential Surface Properties/Extrinsic curvatures

#ifndef PRINCIPAL_CURVATURES
#define PRINCIPAL_CURVATURES

vec2 principal_curvatures(in vec3 gradient, in mat3 hessian)
{
    vec3 normal = normalize(gradient);

    // Generate a stable orthogonal vector to the normal
    vec3 orthogonal = cross(normal, abs(normal.x) < abs(normal.z) ? vec3(1, 0, 0) : vec3(0, 1, 0));

    // Compute arbitrary orthonormal tangent space
    vec3 t0 = normalize(orthogonal);
    vec3 t1 = cross(normal, t0);

    // Build tangent basis matrix (2x3)
    mat2x3 tangents = mat2x3(t0, t1);

    // Project Hessian into tangent space to get the 2x2 shape operator
    mat2 shape = (transpose(tangents) * hessian * tangents) / length(gradient);

    // Compute eigenvalues of the 2x2 shape matrix
    float trace = shape[0][0] + shape[1][1];
    float determinant = determinant(shape);
    float discriminant = sqrt(max(trace * trace - 4.0 * determinant, 0.0));

    // Principal curvatures
    vec2 curvatures = vec2(trace - discriminant, trace + discriminant) * 0.5;
    return curvatures;
}

vec2 principal_curvatures(in vec3 gradient, in mat3 hessian, out vec3 eigenvectors[2])
{
    vec3 normal = normalize(gradient);

    // Generate a stable orthogonal vector to the normal
    vec3 orthogonal = cross(normal, abs(normal.x) < abs(normal.z) ? vec3(1, 0, 0) : vec3(0, 1, 0));

    // Compute arbitrary orthonormal tangent space
    vec3 t0 = normalize(orthogonal);
    vec3 t1 = cross(normal, t0);

    // Build tangent basis matrix (2x3)
    mat2x3 tangents = mat2x3(t0, t1);

    // Project Hessian into tangent space to get the 2x2 shape operator
    mat2 shape = (transpose(tangents) * hessian * tangents) / length(gradient);

    // Compute eigenvalues of the 2x2 shape matrix
    float trace = shape[0][0] + shape[1][1];
    float determinant = determinant(shape);
    float discriminant = sqrt(max(trace * trace - 4.0 * determinant, 0.0));

    // Principal curvatures
    vec2 curvatures = vec2(trace - discriminant, trace + discriminant) * 0.5;

    // compute principal curvature eigenvectors
    float difference = shape[1][1] - shape[0][0];
    eigenvectors[0] = curvatures.x * t0 + (curvatures.x + difference) * t1;
    eigenvectors[1] = curvatures.y * t0 + (curvatures.y + difference) * t1;

    // return curvatures
    return curvatures;
}

#endif