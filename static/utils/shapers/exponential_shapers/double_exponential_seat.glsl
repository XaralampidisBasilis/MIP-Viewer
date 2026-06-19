//https://www.flong.com/archive/texts/code/shapers_exp/

#ifndef DOUBLE_EXPONENTIAL_SEAT
#define DOUBLE_EXPONENTIAL_SEAT

float double_exponential_seat (float x, float a)
{
  float epsilon = 0.00001;
  float min_param_a = 0.0 + epsilon;
  float max_param_a = 1.0 - epsilon;
  a = min(max_param_a, max(min_param_a, a)); 

  float y = 0;
  if (x<=0.5){
    y = (pow(2.0*x, 1-a))/2.0;
  } else {
    y = 1.0 - (pow(2.0*(1.0-x), 1-a))/2.0;
  }
  return y;
}

#endif
