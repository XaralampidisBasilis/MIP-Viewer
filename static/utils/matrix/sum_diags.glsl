#ifndef SUM_DIAGS
#define SUM_DIAGS

// mat2: 2x2 → 3 diagonals
void sum_diags(in mat2 M, out vec3 v) 
{
    v[0] = M[1][0];            // i - j = -1
    v[1] = M[0][0] + M[1][1];  // i - j =  0
    v[2] = M[0][1];            // i - j = +1
}

// mat3: 3x3 → 5 diagonals
void sum_diags(in mat3 M, out float v[5]) 
{
    v[0] = M[2][0];                     // i - j = -2
    v[1] = M[1][0] + M[2][1];           // i - j = -1
    v[2] = M[0][0] + M[1][1] + M[2][2]; // i - j =  0
    v[3] = M[0][1] + M[1][2];           // i - j = +1
    v[4] = M[0][2];                     // i - j = +2
}

// mat4: 4x4 → 7 diagonals
void sum_diags(in mat4 M, out float v[7]) 
{
    v[0] = M[3][0];                                // i - j = -3
    v[1] = M[2][0] + M[3][1];                      // i - j = -2
    v[2] = M[1][0] + M[2][1] + M[3][2];            // i - j = -1
    v[3] = M[0][0] + M[1][1] + M[2][2] + M[3][3];  // i - j =  0
    v[4] = M[0][1] + M[1][2] + M[2][3];            // i - j = +1
    v[5] = M[0][2] + M[1][3];                      // i - j = +2
    v[6] = M[0][3];                                // i - j = +3
}

// mat3x2: 3 cols, 2 rows → 4 diagonals
void sum_diags(in mat3x2 M, out vec4 v) 
{
    v[0] = M[2][0];            // i - j = -2
    v[1] = M[1][0] + M[2][1];  // i - j = -1
    v[2] = M[0][0] + M[1][1];  // i - j =  0
    v[3] = M[0][1];            // i - j = +1
}

// mat2x3: 2 cols, 3 rows → 4 diagonals
void sum_diags(in mat2x3 M, out vec4 v) 
{
    v[0] = M[1][0];            // i - j = -1
    v[1] = M[0][0] + M[1][1];  // i - j =  0
    v[2] = M[0][1] + M[1][2];  // i - j = +1
    v[3] = M[0][2];            // i - j = +2
}

// mat2x4: 2 cols, 4 rows → 5 diagonals
void sum_diags(in mat2x4 M, out float v[5]) 
{
    v[0] = M[1][0];            // i - j = -1
    v[1] = M[0][0] + M[1][1];  // i - j =  0
    v[2] = M[0][1] + M[1][2];  // i - j = +1
    v[3] = M[0][2] + M[1][3];  // i - j = +2
    v[4] = M[0][3];            // i - j = +3
}

// mat4x2: 4 cols, 2 rows → 5 diagonals
void sum_diags(in mat4x2 M, out float v[5]) 
{
    v[0] = M[3][0];            // i - j = -3
    v[1] = M[2][0] + M[3][1];  // i - j = -2
    v[2] = M[1][0] + M[2][1];  // i - j = -1
    v[3] = M[0][0] + M[1][1];  // i - j =  0
    v[4] = M[0][1];            // i - j = +1
}

// mat3x4: 3 cols, 4 rows → 6 diagonals
void sum_diags(in mat3x4 M, out float v[6]) 
{
    v[0] = M[2][0];                      // i - j = -2
    v[1] = M[1][0] + M[2][1];            // i - j = -1
    v[2] = M[0][0] + M[1][1] + M[2][2];  // i - j =  0
    v[3] = M[0][1] + M[1][2] + M[2][3];  // i - j = +1
    v[4] = M[0][2] + M[1][3];            // i - j = +2
    v[5] = M[0][3];                      // i - j = +3
}

// mat4x3: 4 cols, 3 rows → 6 diagonals
void sum_diags(in mat4x3 M, out float v[6]) 
{
    v[0] = M[3][0];                      // i - j = -3
    v[1] = M[2][0] + M[3][1];            // i - j = -2
    v[2] = M[1][0] + M[2][1] + M[3][2];  // i - j = -1
    v[3] = M[0][0] + M[1][1] + M[2][2];  // i - j =  0
    v[4] = M[0][1] + M[1][2];            // i - j = +1
    v[5] = M[0][2];                      // i - j = +2
}

#endif
