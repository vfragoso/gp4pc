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

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <math/solve_gp4pc_polynomial.h>

namespace msft {
namespace {

Eigen::VectorXd BuildMonomialVector(const Eigen::Vector4d& depths) {
  Eigen::VectorXd monomials(15);
  monomials(0) = depths[0] * depths[0];
  monomials(1) = depths[1] * depths[1];
  monomials(2) = depths[2] * depths[2];
  monomials(3) = depths[3] * depths[3];
  monomials(4) = depths[0] * depths[1];
  monomials(5) = depths[0] * depths[2];
  monomials(6) = depths[0] * depths[3];
  monomials(7) = depths[1] * depths[2];
  monomials(8) = depths[1] * depths[3];
  monomials(9) = depths[2] * depths[3];
  monomials(10) = depths[0];
  monomials(11) = depths[1];
  monomials(12) = depths[2];
  monomials(13) = depths[3];
  monomials(14) = 1.0;
  return monomials;
}

TEST(HyperQuadricsPolySolverTests, BasicTest) {
  using Eigen::MatrixXcd;
  using Eigen::MatrixXd;
  using Eigen::Vector4d;
  using Eigen::VectorXd;

  MatrixXd coeff_mat(4, 15);
  coeff_mat << 4.5384, 3.5384, 0, 0, -3.8036, -6.6018, 7.4593, 8.6448, -5.1151,
      0, -53.4298, -36.7695, -16.2069, -17.0343, 438.2425,
      0, 0, -9.8376, -8.8376, 0, 3.0456, -3.8306, -3.1094, 2.0480,
      16.5214, 10.4637, 2.3920, -8.9595, 47.8821, -191.6141,
      1.0000, 1.0000, -5.0307, -5.0307, -0.9419, 0, 0, 0, 0, 8.9010, -7.3813,
      -8.3257, -8.1825, 22.5721, 1.4279,
      -0.7478, 1.0000, -1.7478, 0, -0.9419, 2.3459, 0, 0, 0, 0, 3.9632, -8.3257,
      1.3945, 0, 18.5446;
  const Vector4d solution(6.7687, 8.0037, 5.2626, 7.2827);

  // Estimate solution.
  const MatrixXcd solutions = SolveGp4pcPolynomial(coeff_mat);

  // Find the closest solution and report the residual error.
  Vector4d estimated_solution;
  Vector4d best_solution;
  double best_sq_error = std::numeric_limits<double>::max();
  for (int i = 0; i < solutions.cols(); ++i) {
    estimated_solution[0] = solutions(0, i).real();
    estimated_solution[1] = solutions(1, i).real();
    estimated_solution[2] = solutions(2, i).real();
    estimated_solution[3] = solutions(3, i).real();
    const double sq_error = (estimated_solution - solution).squaredNorm();
    if (best_sq_error > sq_error) {
      best_sq_error = sq_error;
      best_solution = estimated_solution;
    }
  }

  // Test estimated solution.
  const Eigen::Vector4d residual = coeff_mat * BuildMonomialVector(best_solution);

  EXPECT_NEAR(residual.squaredNorm(), 0.0, 1e-4);
}

}  // namespace
}  // namespace msft
