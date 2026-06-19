//https://www.flong.com/archive/texts/code/shapers_circ/

#ifndef DOUBLE_CIRCLE_SEAT
#define DOUBLE_CIRCLE_SEAT

float double_circle_seat (float x, float a)
{
  float min_param_a = 0.0;
  float max_param_a = 1.0;
  a = max(min_param_a, min(max_param_a, a)); 

  float y = 0;
  if (x<=a){
    y = sqrt(sq(a) - sq(x-a));
  } else {
    y = 1 - sqrt(sq(1-a) - sq(x-a));
  }
  return y;
}

#endif