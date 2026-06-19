
#ifndef SPLIT_BERNSTEIN_SIGN_CHANGE
#define SPLIT_BERNSTEIN_SIGN_CHANGE

#ifndef SPLIT_BERNSTEIN
#include "./split_bernstein"
#endif
#ifndef SIGN_CHANGE
#include "./sign_change"
#endif

// bool split_bernstein_sign_change(in vec3 b) 
// {
//     vec3 r1, r2, r3;

//     split_bernstein(b, b, r1); // [0,0.5], [0.5,1]
//     split_bernstein(b, b, r2); // [0,0.25], [0.25,0.5]
//     split_bernstein(b, b, r3);  if (sign_change(b) || sign_change(r3)) return true;
//     split_bernstein(r2, b, r3); if (sign_change(b) || sign_change(r3)) return true;

//     split_bernstein(r1, b, r2); // [0.5,0.75], [0.75,1.0]
//     split_bernstein(b, b, r3);  if (sign_change(b) || sign_change(r3)) return true;
//     split_bernstein(r2, b, r3); if (sign_change(b) || sign_change(r3)) return true;

//     return false;
// }

// bool split_bernstein_sign_change(in vec4 b) 
// {
//     vec4 r1, r2, r3;

//     split_bernstein(b, b, r1); // [0,0.5], [0.5,1]
//     split_bernstein(b, b, r2); // [0,0.25], [0.25,0.5]
//     split_bernstein(b, b, r3);  if (sign_change(b) || sign_change(r3)) return true;
//     split_bernstein(r2, b, r3); if (sign_change(b) || sign_change(r3)) return true;

//     split_bernstein(r1, b, r2); // [0.5,0.75], [0.75,1.0]
//     split_bernstein(b, b, r3);  if (sign_change(b) || sign_change(r3)) return true;
//     split_bernstein(r2, b, r3); if (sign_change(b) || sign_change(r3)) return true;

//     return false;
// }

// bool split_bernstein_sign_change(in float b[5]) 
// {
//     float r1[5], r2[5], r3[5];

//     split_bernstein(b, b, r1); // [0,0.5], [0.5,1]
//     split_bernstein(b, b, r2); // [0,0.25], [0.25,0.5]
//     split_bernstein(b, b, r3);  if (sign_change(b) || sign_change(r3)) return true;
//     split_bernstein(r2, b, r3); if (sign_change(b) || sign_change(r3)) return true;

//     split_bernstein(r1, b, r2); // [0.5,0.75], [0.75,1.0]
//     split_bernstein(b, b, r3);  if (sign_change(b) || sign_change(r3)) return true;
//     split_bernstein(r2, b, r3); if (sign_change(b) || sign_change(r3)) return true;

//     return false;
// }


// bool split_bernstein_sign_change(in float b[6]) 
// {
//     float r1[6], r2[6], r3[6];

//     split_bernstein(b, b, r1); // [0,0.5], [0.5,1]
//     split_bernstein(b, b, r2); // [0,0.25], [0.25,0.5]
//     split_bernstein(b, b, r3);  if (sign_change(b) || sign_change(r3)) return true;
//     split_bernstein(r2, b, r3); if (sign_change(b) || sign_change(r3)) return true;

//     split_bernstein(r1, b, r2); // [0.5,0.75], [0.75,1.0]
//     split_bernstein(b, b, r3);  if (sign_change(b) || sign_change(r3)) return true;
//     split_bernstein(r2, b, r3); if (sign_change(b) || sign_change(r3)) return true;

//     return false;
// }

bool split_bernstein_sign_change(in float b[6]) 
{
    float r[6];

    split_bernstein(b, b, r); // [0,0.5], [0.5,1]

    return sign_change(b) || sign_change(r);
}



#endif