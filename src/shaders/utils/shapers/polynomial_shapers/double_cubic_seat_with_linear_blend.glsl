//https://www.flong.com/archive/texts/code/shapers_poly/

#ifndef DOUBLE_CUBIC_SEAT_WITH_LINEAR_BLEND
#define DOUBLE_CUBIC_SEAT_WITH_LINEAR_BLEND

float double_cubic_seat_with_linear_blend (float x, float a, float b)
{
  float epsilon = 0.00001;
  float min_param_a = 0.0 + epsilon;
  float max_param_a = 1.0 - epsilon;
  float min_param_b = 0.0;
  float max_param_b = 1.0;
  a = min(max_param_a, max(min_param_a, a));  
  b = min(max_param_b, max(min_param_b, b)); 
  b = 1.0 - b; //reverse for intelligibility.
  
  float y = 0;
  if (x<=a)
  {
    y = b*x + (1-b)*a*(1-pow(1-x/a, 3.0));
  } 
  else 
  {
    y = b*x + (1-b)*(a + (1-a)*pow((x-a)/(1-a), 3.0));
  }

  return y;

}

#endif
