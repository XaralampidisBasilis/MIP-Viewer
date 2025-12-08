//https://www.flong.com/archive/texts/code/shapers_poly/

#ifndef QUADRATIC_THROUGH_A_GIVEN_POINT
#define QUADRATIC_THROUGH_A_GIVEN_POINT

float quadratic_through_a_given_point (float x, float a, float b)
{  
  float epsilon = 0.00001;
  float min_param_a = 0.0 + epsilon;
  float max_param_a = 1.0 - epsilon;
  float min_param_b = 0.0;
  float max_param_b = 1.0;
  a = min(max_param_a, max(min_param_a, a));  
  b = min(max_param_b, max(min_param_b, b)); 
  
  float A = (1-b)/(1-a) - (b/a);
  float B = (A*(a*a)-b)/a;
  float y = A*(x*x) - B*(x);
  y = min(1,max(0,y)); 
  
  return y;
}

#endif