// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Generalized Pose-and-Scale Estimation using 4-Point Congruence Constraints
//
// Victor Fragoso and Sudipta Sinha.
// In Proc. of the IEEE International Conf. on 3D Vision (3DV) 2020.
//
// Please contact the author of this library if you have any questions.
// Author: Victor Fragoso (victor.fragoso@microsoft.com)

#ifndef GP4PC_MATH_UTILS_H_
#define GP4PC_MATH_UTILS_H_
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace msft {

// Convert degrees to radians.
constexpr double DegToRad(double angle_degrees) noexcept {
  constexpr double kDegToRad = M_PI / 180.0;
  return angle_degrees * kDegToRad;
}

// Convert radiants to degrees.
constexpr double RadToDeg(const double angle_radians) noexcept {
  constexpr double kRadToDeg = 180.0 / M_PI;
  return angle_radians * kRadToDeg;
}

}  // namespace msft

#endif  // GP4PC_MATH_UTILS_H_

