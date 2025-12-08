// From the paper "Real-Time Ray-Casting and Advanced Shading of Discrete Isosurfaces"
// in the 4.1. Differential Surface Properties/Extrinsic curvatures

#ifndef COMPUTE_CURVATURES
#define COMPUTE_CURVATURES

vec2 compute_curvatures(in vec3 gradient, in mat3 hessian)
{
    vec3 n = normalize(gradient);
    vec3 t0 = normalize(cross(n, abs(n.z) < 0.999 ? vec3(0,0,1) : vec3(0,1,0)));
    vec3 t1 = cross(n, t0);

    mat2x3 T = mat2x3(t0, t1);
    mat2 A = transpose(T) * hessian * T / length(gradient);

    float traceA = A[0][0] + A[1][1];
    float detA = A[0][0]*A[1][1] - A[0][1]*A[1][0];
    float disc = sqrt(max(traceA*traceA - 4.0*detA, 0.0));

    vec2 kappa = 0.5 * vec2(traceA - disc, traceA + disc);

    return kappa;
}

vec2 compute_curvatures(in vec3 gradient, in mat3 hessian, out vec3 eigenvectors[2])
{
    vec3 n = normalize(gradient);
    vec3 t0 = normalize(cross(n, abs(n.z) < 0.999 ? vec3(0,0,1) : vec3(0,1,0)));
    vec3 t1 = cross(n, t0);

    mat2x3 T = mat2x3(t0, t1);
    mat2 A = transpose(T) * hessian * T / length(gradient);

    float traceA = A[0][0] + A[1][1];
    float detA = A[0][0]*A[1][1] - A[0][1]*A[1][0];
    float disc = sqrt(max(traceA*traceA - 4.0*detA, 0.0));

    vec2 kappa = 0.5 * vec2(traceA - disc, traceA + disc);

    eigenvectors[0] = normalize(kappa.x * t0 + (kappa.x + A[1][1] - A[0][0]) * t1);
    eigenvectors[1] = normalize(kappa.y * t0 + (kappa.y + A[1][1] - A[0][0]) * t1);

    return kappa;
}

#endif