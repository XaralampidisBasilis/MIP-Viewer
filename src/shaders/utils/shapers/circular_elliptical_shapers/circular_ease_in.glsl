//https://www.flong.com/archive/texts/code/shapers_circ/

#ifndef CIRCULAR_EASE_IN
#define CIRCULAR_EASE_IN

float circular_ease_in (float x)
{
  float y = 1 - sqrt(1 - x*x);
  return y;
}

#endif