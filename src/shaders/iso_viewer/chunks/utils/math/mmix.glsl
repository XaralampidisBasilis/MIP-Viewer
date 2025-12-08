/*
contributors: Patricio Gonzalez Vivo
description: expands mix to linearly mix more than two values
use: <float|vec2|vec3|vec4> mmix(<float|vec2|vec3|vec4> a, <float|vec2|vec3|vec4> b, <float|vec2|vec3|vec4> c [, <float|vec2|vec3|vec4> d], <float> pct)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef MMIX
#define MMIX

int    mmix(in int    a, in int    b, in int    pct) { return a + (b - a) * pct; }
ivec2  mmix(in ivec2  a, in ivec2  b, in int    pct) { return a + (b - a) * pct; }
ivec3  mmix(in ivec3  a, in ivec3  b, in int    pct) { return a + (b - a) * pct; }
ivec4  mmix(in ivec4  a, in ivec4  b, in int    pct) { return a + (b - a) * pct; }
ivec2  mmix(in ivec2  a, in ivec2  b, in ivec2  pct) { return a + (b - a) * pct; }
ivec3  mmix(in ivec3  a, in ivec3  b, in ivec3  pct) { return a + (b - a) * pct; }
ivec4  mmix(in ivec4  a, in ivec4  b, in ivec4  pct) { return a + (b - a) * pct; }

float mmix(in float a, in float b, in float pct) { return mix(a, b, pct); }
vec2  mmix(in vec2  a, in vec2  b, in float pct) { return mix(a, b, pct); }
vec3  mmix(in vec3  a, in vec3  b, in float pct) { return mix(a, b, pct); }
vec4  mmix(in vec4  a, in vec4  b, in float pct) { return mix(a, b, pct); }
vec2  mmix(in vec2  a, in vec2  b, in vec2  pct) { return mix(a, b, pct); }
vec3  mmix(in vec3  a, in vec3  b, in vec3  pct) { return mix(a, b, pct); }
vec4  mmix(in vec4  a, in vec4  b, in vec4  pct) { return mix(a, b, pct); }

float mmix(in float a, in float b, in float c, in float pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec2 mmix(vec2 a, vec2 b, vec2 c, float pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec2 mmix(vec2 a, vec2 b, vec2 c, vec2 pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec3 mmix(vec3 a, vec3 b, vec3 c, float pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec3 mmix(vec3 a, vec3 b, vec3 c, vec3 pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec4 mmix(vec4 a, vec4 b, vec4 c, float pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

vec4 mmix(vec4 a, vec4 b, vec4 c, vec4 pct) {
    return mix(
        mix(a, b, 2. * pct),
        mix(b, c, 2. * (max(pct, .5) - .5)),
        step(.5, pct)
    );
}

#endif // MMIX