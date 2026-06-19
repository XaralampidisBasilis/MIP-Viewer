//https://www.flong.com/archive/texts/code/shapers_poly/

#ifndef DOUBLE_CUBIC_SEAT
#define DOUBLE_CUBIC_SEAT

float double_cubic_seat (float x, float a, float b)
{
  
  float epsilon = 0.00001;
  float min_param_a = 0.0 + epsilon;
  float max_param_a = 1.0 - epsilon;
  float min_param_b = 0.0;
  float max_param_b = 1.0;
  a = min(max_param_a, max(min_param_a, a));  
  b = min(max_param_b, max(min_param_b, b)); 
  
  float y = 0;
  if (x <= a)
  {
    y = b - b*pow(1-x/a, 3.0);
  } 
  else 
  {
    y = b + (1-b)*pow((x-a)/(1-a), 3.0);
  }
  return y;
}

#endif
