//https://www.flong.com/archive/texts/code/shapers_circ/

#ifndef DOUBLE_ELLIPTIC_SIGMOID
#define DOUBLE_ELLIPTIC_SIGMOID

float double_elliptic_sigmoid (float x, float a, float b)
{
  float epsilon = 0.00001;
  float min_param_a = 0.0 + epsilon;
  float max_param_a = 1.0 - epsilon;
  float min_param_b = 0.0;
  float max_param_b = 1.0;
  a = max(min_param_a, min(max_param_a, a)); 
  b = max(min_param_b, min(max_param_b, b));
 
  float y = 0;
  if (x<=a){
    y = b * (1 - (sqrt(sq(a) - sq(x))/a));
  } else {
    y = b + ((1-b)/(1-a))*sqrt(sq(1-a) - sq(x-1));
  }
  return y;
}

#endif