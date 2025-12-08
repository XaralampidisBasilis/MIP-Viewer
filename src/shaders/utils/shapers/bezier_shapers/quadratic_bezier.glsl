//https://www.flong.com/archive/texts/code/shapers_bez/

#ifndef QUADRATIC_BEZIER
#define QUADRATIC_BEZIER

float quadratic_bezier (float x, float a, float b)
{
  // adapted from BEZMATH.PS (1993)
  // by Don Lancaster, SYNERGETICS Inc. 
  // http://www.tinaja.com/text/bezmath.html

  float epsilon = 0.00001;
  a = max(0, min(1, a)); 
  b = max(0, min(1, b)); 
  if (a == 0.5){
    a += epsilon;
  }
  
  // solve t from x (an inverse operation)
  float om2a = 1 - 2*a;
  float t = (sqrt(a*a + om2a*x) - a)/om2a;
  float y = (1-2*b)*(t*t) + (2*b)*t;
  return y;
}

#endif
