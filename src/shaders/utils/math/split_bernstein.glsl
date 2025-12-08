
#ifndef SPLIT_BERNSTEIN
#define SPLIT_BERNSTEIN

// Subdivide a Bernstein polynomial at t = 0.5 using de Casteljau's algorithm.
// This produces two new sets of Bernstein coefficients representing the polynomial on [0, 0.5] and [0.5, 1].

void split_bernstein(in vec3 b, out vec3 left, out vec3 right)
{
    // Stage 1
    vec2 m = mix(b.xy, b.yz, 0.5); // m = [m0, m1]

    // Second-level midpoint
    float n = mix(m.x, m.y, 0.5); // n0

    // Left [0, 0.5]
    left = vec3(b.x, m.x, n);

    // Right  [0.5, 1]
    right = vec3(n, m.y, b.z);
}

void split_bernstein(in vec4 b, out vec4 left, out vec4 right) 
{
    // Stage 1
    vec3 m = mix(b.xyz, b.yzw, 0.5); 

    // Stage 2
    vec2 n = mix(m.xy, m.yz, 0.5);   

    // Stage 3
    float p = mix(n.x, n.y, 0.5);

    // Left [0, 0.5]
    left = vec4(b.x, m.x, n.x, p);   

    // Right  [0.5, 1]
    right = vec4(p, n.y, m.z, b.w);  
}

void split_bernstein(in float b[5], out float left[5], out float right[5]) 
{
    // Stage 0
    vec4 b0 = vec4(b[0], b[1], b[2], b[3]);
    vec4 b1 = vec4(b[1], b[2], b[3], b[4]);

    // Stage 1
    vec4 m = mix(b0, b1, 0.5);

    // Stage 2
    vec3 n = mix(m.xyz, m.yzw, 0.5);

    // Stage 3
    vec2 o = mix(n.xy, n.yz, 0.5);

    // Stage 4
    float p = mix(o.x, o.y, 0.5);

    // Left [0, 0.5]
    left[0] = b[0];
    left[1] = m.x;
    left[2] = n.x;
    left[3] = o.x;
    left[4] = p;

    // Right [0.5, 1]
    right[0] = p;
    right[1] = o.y;
    right[2] = n.z;
    right[3] = m.w;
    right[4] = b[4];
}

void split_bernstein(in float b[6], out float left[6], out float right[6]) 
{
    // Stage 1
    float m0 = (b[0] + b[1]) * 0.5;
    float m1 = (b[1] + b[2]) * 0.5;
    float m2 = (b[2] + b[3]) * 0.5;
    float m3 = (b[3] + b[4]) * 0.5;
    float m4 = (b[4] + b[5]) * 0.5;

    // Stage 2
    vec4 n = vec4(
        (m0 + m1) * 0.5,
        (m1 + m2) * 0.5,
        (m2 + m3) * 0.5,
        (m3 + m4) * 0.5
    );  

    // Stage 3
    vec3 o = mix(n.xyz, n.yzw, 0.5);

    // Stage 4
    vec2 p = mix(o.xy, o.yz, 0.5);

    // Stage 5
    float q = mix(p.x, p.y, 0.5);

    // Left [0, 0.5]
    left[0] = b[0];
    left[1] = m0;
    left[2] = n.x;
    left[3] = o.x;
    left[4] = p.x;
    left[5] = q;

    // Right [0.5, 1.0]
    right[0] = q;
    right[1] = p.y;
    right[2] = o.z;
    right[3] = n.w;
    right[4] = m4;
    right[5] = b[5];
}


#endif