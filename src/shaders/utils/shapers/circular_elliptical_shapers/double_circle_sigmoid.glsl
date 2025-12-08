//https://www.flong.com/archive/texts/code/shapers_circ/

#ifndef DOUBLE_CIRCLE_SIGMOID
#define DOUBLE_CIRCLE_SIGMOID

float double_circle_sigmoid (float x, float a)
{
  float min_param_a = 0.0;
  float max_param_a = 1.0;
  a = max(min_param_a, min(max_param_a, a)); 

  float y = 0;
  if (x<=a){
    y = a - sqrt(a*a - x*x);
  } else {
    y = a + sqrt(sq(1-a) - sq(x-1));
  }
  return y;
}

#endif