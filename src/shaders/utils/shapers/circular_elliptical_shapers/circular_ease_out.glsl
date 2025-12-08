//https://www.flong.com/archive/texts/code/shapers_circ/

#ifndef CIRCULAR_EASE_OUT
#define CIRCULAR_EASE_OUT

float circular_ease_out (float x)
{
  float y = sqrt(1 - sq(1 - x));
  return y;
}

#endif