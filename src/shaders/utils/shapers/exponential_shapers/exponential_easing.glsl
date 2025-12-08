//https://www.flong.com/archive/texts/code/shapers_exp/

#ifndef EXPONENTIAL_EASING
#define EXPONENTIAL_EASING

float exponential_easing (float x, float a)
{  
  float epsilon = 0.00001;
  float min_param_a = 0.0 + epsilon;
  float max_param_a = 1.0 - epsilon;
  a = max(min_param_a, min(max_param_a, a));
  
  if (a < 0.5){
    // emphasis
    a = 2.0*(a);
    float y = pow(x, a);
    return y;
  } else {
    // de-emphasis
    a = 2.0*(a-0.5);
    float y = pow(x, 1.0/(1-a));
    return y;
  }
}

#endif