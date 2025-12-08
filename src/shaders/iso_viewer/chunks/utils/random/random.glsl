/*
contributors: ["Patricio Gonzalez Vivo", "David Hoskins", "Inigo Quilez"]
description: Pass a value and get some random normalize value between 0 and 1
use: float random[2|3](<float|vec2|vec3> value)
license:
    - MIT License (MIT) Copyright 2014, David Hoskins
*/

#ifndef RANDOM
#define RANDOM

#define RANDOM_SCALE vec4(443.897, 441.423, .0973, .1099)

float random(in vec3 pos) {
    pos  = fract(pos * RANDOM_SCALE.xyz);
    pos += dot(pos, pos.zyx + 31.32);
    return fract((pos.x + pos.y) * pos.z);
}

#endif // RANDOM
