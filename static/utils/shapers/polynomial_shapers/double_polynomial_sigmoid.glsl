//https://www.flong.com/archive/texts/code/shapers_poly/

#ifndef DOUBLE_POLYNOMIAL_SIGMOID
#define DOUBLE_POLYNOMIAL_SIGMOID

float double_polynomial_sigmoid(float x, float a, float b, int n)
{
  float y = 0;

  if (n%2 == 0)
  { 
    // even polynomial
    if (x<=0.5)
    {
      y = pow(2.0*x, n)/2.0;
    } 
    else 
    {
      y = 1.0 - pow(2*(x-1), n)/2.0;
    }
  } 
  else 
  { 
    // odd polynomial
    if (x<=0.5)
    {
      y = pow(2.0*x, n)/2.0;
    } 
    else 
    {
      y = 1.0 + pow(2.0*(x-1), n)/2.0;
    }
  }

  return y;
}

#endif
