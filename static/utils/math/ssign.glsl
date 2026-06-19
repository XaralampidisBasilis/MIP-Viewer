#ifndef SSIGN
#define SSIGN

#ifndef PICK
#include "./pick"
#endif

float ssign(in float v) 
{ 
    return pick(v < 0.0, -1.0, 1.0);
}

vec2 ssign(in vec2 v) 
{ 
    return pick(
        lessThan(v, vec2(0.0)), 
        vec2(-1.0), 
        vec2( 1.0)
    ); 
}

vec3 ssign(in vec3 v) 
{   
    return pick(
        lessThan(v, vec3(0.0)), 
        vec3(-1.0), 
        vec3( 1.0)
    );
}

vec4 ssign(in vec4 v) 
{ 
    return pick(
        lessThan(v, vec4(0.0)), 
        vec4(-1.0), 
        vec4( 1.0)
    ); 
}

#endif // SSIGN
