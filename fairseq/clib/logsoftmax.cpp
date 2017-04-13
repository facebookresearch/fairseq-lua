/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cfloat>
#include <cmath>

extern "C" {

/* Credits to Leon Bottou */
double approxexpminus(const double x)
{
  /* fast approximation of exp(-x) for x positive */
  const double a0 = 1.0;
  const double a1 = 0.125;
  const double a2 = 0.0078125;
  const double a3 = 0.00032552083;
  const double a4 = 1.0172526e-5;
  if (x < 13.0)
  {
    double y;
    y = a0+x*(a1+x*(a2+x*(a3+x*a4)));
    y *= y;
    y *= y;
    y *= y;
    y = 1/y;
    return y;
  }
  return 0;
}

void logsoftmax1d(float* input, float* output, int sz1) {
  float max = -FLT_MAX;
  float* in = input;
  for (int i = 0; i < sz1; i++) {
    float v = *in++;
    if (max < v) {max = v;}
  }

  double logsum = 0;
  in = input;
  for (int i = 0; i < sz1; i++) {
    logsum += approxexpminus(max - *in++);
  }
  logsum = max + log(logsum);

  for (int i = 0; i < sz1; i++) {
    *output++ = *input++ - logsum;
  }
}

void logsoftmax2d(float* input, float* output, int sz1, int sz2) {
  #pragma omp parallel for
  for (int i = 0; i < sz1; i++) {
    logsoftmax1d(input + i * sz2, output + i * sz2, sz2);
  }
}

}
