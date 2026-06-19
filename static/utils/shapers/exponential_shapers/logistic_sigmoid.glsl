//https://www.flong.com/archive/texts/code/shapers_exp/

#ifndef LOGISTIC_SIGMOID
#define LOGISTIC_SIGMOID

float logistic_sigmoid (float x, float a)
{
  // n.b.: this Logistic Sigmoid has been normalized.

  float epsilon = 0.0001;
  float min_param_a = 0.0 + epsilon;
  float max_param_a = 1.0 - epsilon;
  a = max(min_param_a, min(max_param_a, a));
  a = (1/(1-a) - 1);

  float A = 1.0 / (1.0 + exp(0 -((x-0.5)*a*2.0)));
  float B = 1.0 / (1.0 + exp(a));
  float C = 1.0 / (1.0 + exp(0-a)); 
  float y = (A-B)/(C-B);
  return y;
}

#endif