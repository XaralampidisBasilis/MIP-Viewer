#ifndef DIFF
#define DIFF

float diff(in vec2 v) { return v.y   - v.x;   }
vec2  diff(in vec3 v) { return v.yz  - v.xy;  }
vec3  diff(in vec4 v) { return v.yzw - v.xyz; }

#endif // DIFF

