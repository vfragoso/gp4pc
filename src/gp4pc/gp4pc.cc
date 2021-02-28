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
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <glog/logging.h>
#include <gp4pc/gp4pc.h>
#include <math/solve_gp4pc_polynomial.h>
#include <third_party/align_point_clouds.h>
#include <vector>

namespace msft {
namespace {

// Default coplanar threshold.
const double kDefaultCoplanarThresh = 1e-3;
const double kDefaultColinearThresh = 1e-2;
const double kRealNumberThreshold = 0.0;
const double kDuplicateSolnErrorThreshold = 1e-6;
const double kDuplicateCameraCenterErrorThreshold = 1e-6;
// Dimensions of the coefficient matrix: kNumConstraints x kNumMonomials.
const int kNumConstraints = 4;
const int kNumMonomials = 15;
// Max. num. of depth solutions.
const int kMaxNumDepthSolutions = 16;
// Size of the minimal sample.
const int kSizeOfMinimalSample = 4;

inline Eigen::Vector3d ComputeCameraPoint(const Eigen::Vector3d& camera_center,
                                          const Eigen::Vector3d& ray_direction,
                                          const double depth) {
  return camera_center + depth * ray_direction;
}

bool IsInputValid(const Gp4pc::Input& input) {
  // Check that the camera centers are different.
  int num_different_origins = 0;
  for (int i = 0; i < input.ray_origins.size(); ++i) {
    for (int j = i + 1; j < input.ray_origins.size(); ++j) {
      const double distance =
          (input.ray_origins[i] - input.ray_origins[j]).squaredNorm();
      if (distance > kDuplicateCameraCenterErrorThreshold) {
        num_different_origins += 1;
      }
    }
  }
  const bool has_different_origins = num_different_origins > 0;
  return has_different_origins;
}

// Build least-squares system Ax = b, where
// b = [p_1 p_2 p_3 p_4]^T
// and
//     | L(w_1)  I |
// A = | L(w_2)  I |
//     | L(w_3)  I |
//     | L(w_4)  I |.
void BuildLinearSystem(const std::vector<Eigen::Vector3d>& world_points,
                       const std::vector<Eigen::Vector3d>& camera_points,
                       Eigen::VectorXd* camera_points_vec_ptr,
                       Eigen::MatrixXd* world_points_mat_ptr) {
  // Dimensionality of the points.
  static const int kPointDimension = 3;

  const int num_points = world_points.size();
  Eigen::VectorXd& camera_points_vec = *camera_points_vec_ptr;
  Eigen::MatrixXd& world_points_mat = *world_points_mat_ptr;
  world_points_mat.setZero();

  int row_index = 0;
  for (int i = 0; i < num_points; ++i) {
    // Fill in camera points.
    camera_points_vec.segment(i * kPointDimension, kPointDimension) =
        camera_points[i];

    // Fill in world-point matrices.
    row_index = i * kPointDimension;
    world_points_mat.block(row_index, 0, 1, 3) = world_points[i].transpose();
    world_points_mat.block(row_index + 1, 3, 1, 3) =
        world_points[i].transpose();
    world_points_mat.block(row_index + 2, 6, 1, 3) =
        world_points[i].transpose();
    world_points_mat.block(row_index, 9, 3, 3) = Eigen::Matrix3d::Identity();
  }
}

bool IsInputDegenerateOrPlanar(const Gp4pc::Input& input,
                               const double coplanar_threshold,
                               const double colinear_threshold,
                               bool* is_planar) {
  *CHECK_NOTNULL(is_planar) = false;
  // Compute the SVD of the input 3D points.
  const Eigen::Vector3d mean_direction =
      (input.world_points[0] + input.world_points[1] + input.world_points[2] +
       input.world_points[3]) /
      4;
  Eigen::Matrix<double, 4, 3> normalized_points;
  normalized_points.row(0) = input.world_points[0] - mean_direction;
  normalized_points.row(1) = input.world_points[1] - mean_direction;
  normalized_points.row(2) = input.world_points[2] - mean_direction;
  normalized_points.row(3) = input.world_points[3] - mean_direction;
  const Eigen::Matrix3d covariance =
      normalized_points.transpose() * normalized_points;
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::Vector3d singular_values = svd.singularValues();

  // Is input degenerate, i.e., a line?
  double singular_value_ratio = singular_values[1] / singular_values[0];
  if (singular_value_ratio < colinear_threshold) {
    VLOG(2) << "Singular value ratio (colinear test): " << singular_value_ratio
            << " threshold: " << colinear_threshold;
    return true;
  }

  // Is the input planar?
  singular_value_ratio = singular_values[2] / singular_values[0];
  *is_planar = singular_value_ratio < coplanar_threshold;
  VLOG_IF(4, *is_planar) << "Singular value ratio (coplanar test): "
                         << singular_value_ratio
                         << " threshold: " << coplanar_threshold;

  return false;
}

// This function computes the values of t1 and t3 given the four points:
// (x1, x2, x3, x4).
//
// The point (x1 + t1 * v1) + t3 * v3 must be equal to the point (x3 + t2 * v2).
// This gives a linear system in t1, t2, t3. Solve it. This gives the following
// closest points on the lines.
//    x1 + t1 * v1;
//    x3 + t2 * v2;
Eigen::Vector2d Findt1t3(const Eigen::Vector3d& x1,
                         const Eigen::Vector3d& x2,
                         const Eigen::Vector3d& x3,
                         const Eigen::Vector3d& x4) {
  const Eigen::Vector3d v1 = x2 - x1;
  const Eigen::Vector3d v2 = x4 - x3;
  const Eigen::Vector3d v3 = v1.cross(v2);
  const Eigen::Vector3d b = x3 - x1;

  // A = [v1 -v2 v3]; (MATLAB code).
  Eigen::MatrixXd A(3,3);
  A.col(0) = v1;
  A.col(1) = -v2;
  A.col(2) = v3;

  // t = A \ b; (MATLAB code).
  const Eigen::VectorXd x = A.lu().solve(b);

  // Solution
  return x;
}

// This function calculates (f1, f2, f3, f4) and (K1, K2) from the four points
// (x1, x2, x3, x4).
// TODO(vfragoso): Pass world points instead of x_i and create a structure
// for f_i and k_j.
void CalculateRatiosFromFourPoints(const Eigen::Vector3d& x1,
                                   const Eigen::Vector3d& x2,
                                   const Eigen::Vector3d& x3,
                                   const Eigen::Vector3d& x4,
                                   double& f1,
                                   double& f2,
                                   double& f3,
                                   double& f4,
                                   double& K1,
                                   double& K2) {
  //
  // x1--x2 is the first line segment and x3--x4 is the second line segment.
  // z1 is on x1--x2, z3 is on x3--x4 and (z1--z3) is also perpendicular
  // to x1--x2 and x3--x4
  //
  // t1 is equal to dist(x1,z1) / dist(x1,x2)
  // t2 is equal to dist(x3,z3) / dist(x3,x4)
  //
  // [t1, t3, x1, x2, x3, x4] = findClosestPoints(X,1,2,3,4); (MATLAB code)
  //

  const Eigen::Vector2d t_vec = Findt1t3(x1, x2, x3, x4);

  const double len12 = (x1 - x2).norm();
  const double len34 = (x3 - x4).norm();
  const double len13 = (x1 - x3).norm();

  const double s12 = len12 * len12;
  const double s34 = len34 * len34;
  const double s13 = len13 * len13;

  K1 = s12 / s34;
  K2 = s12 / s13;

  f1 = 1 - t_vec(0);
  f2 = t_vec(0);
  f3 = 1 - t_vec(1);
  f4 = t_vec(1);
}

// TODO(vfragoso): Polish this function.
Eigen::MatrixXd SolveForDepthsFromAPlanarInput(const Gp4pc::Input& input) {
  // Useful aliases.
  // Camera centers.
  const std::vector<Eigen::Vector3d>& ray_origins = input.ray_origins;
  // Unit rays originating from ray_origins.
  const std::vector<Eigen::Vector3d>& ray_directions = input.ray_directions;
  // World points.
  const std::vector<Eigen::Vector3d>& world_points = input.world_points;

  // Verify the input is consistent.
  CHECK_EQ(ray_origins.size(), ray_directions.size());
  CHECK_EQ(ray_origins.size(), world_points.size());

  // The mathematical relationship is the following:
  //
  // ray_origins[i] + depth[i] * ray_directions[i] =
  //         s * R * world_points[i] + t,
  //
  // where s is scale, R is a rotation matrix, and t is translation.

  // Input Constants.
  const Eigen::Vector3d& p1 = ray_origins[0];
  const Eigen::Vector3d& p2 = ray_origins[1];
  const Eigen::Vector3d& p3 = ray_origins[2];
  const Eigen::Vector3d& p4 = ray_origins[3];

  // Input Constants.
  const Eigen::Vector3d& u1 = ray_directions[0];
  const Eigen::Vector3d& u2 = ray_directions[1];
  const Eigen::Vector3d& u3 = ray_directions[2];
  const Eigen::Vector3d& u4 = ray_directions[3];

  // Input Constants.
  const Eigen::Vector3d& x1 = world_points[0];
  const Eigen::Vector3d& x2 = world_points[1];
  const Eigen::Vector3d& x3 = world_points[2];
  const Eigen::Vector3d& x4 = world_points[3];

  // Derived Constants.
  double f1, f2, f3, f4, K1, K2;
  CalculateRatiosFromFourPoints(x1, x2, x3, x4, f1, f2, f3, f4, K1, K2);

  // Derived Constants.
  const Eigen::Vector3d p5 = f1 * p1 + f2 * p2 - f3 * p3 - f4 * p4;
  const Eigen::Vector3d p6 = p1 - p2;
  const Eigen::Vector3d p8 = p1 - p3;

  // Derived Constants.
  const Eigen::Vector3d v1 = f1 * u1;
  const Eigen::Vector3d v2 = f2 * u2;
  const Eigen::Vector3d v3 = -f3 * u3;
  const Eigen::Vector3d v4 = -f4 * u4;

  // Think of the 3 x 3 system in s1, s2, s3
  // [a1 b1 c1]   [d1]
  // [a2 b2 c2] = [d2]
  // [a3 b3 c3] = [d3]
  double a1 = v1(0);
  double b1 = v2(0);
  double c1 = v3(0);
  double a2 = v1(1);
  double b2 = v2(1);
  double c2 = v3(1);
  double a3 = v1(2);
  double b3 = v2(2);
  double c3 = v3(2);

  double denom = a1 * b2 * c3 + b1 * c2 * a3 + c1 * a2 * b3 - a3 * b2 * c1 -
                 b3 * c2 * a1 - c3 * a2 * b1;

  double Coef11 = b2 * c3 - b3 * c2;
  double Coef12 = c1 * b3 - c3 * b1;
  double Coef13 = b1 * c2 - b2 * c1;

  double G1 = (-v4(0) * Coef11 - v4(1) * Coef12 - v4(2) * Coef13) / denom;
  double H1 = (-p5(0) * Coef11 - p5(1) * Coef12 - p5(2) * Coef13) / denom;

  double Coef21 = c2 * a3 - c3 * a2;
  double Coef22 = a1 * c3 - a3 * c1;
  double Coef23 = c1 * a2 - c2 * a1;

  double G2 = (-v4(0) * Coef21 - v4(1) * Coef22 - v4(2) * Coef23) / denom;
  double H2 = (-p5(0) * Coef21 - p5(1) * Coef22 - p5(2) * Coef23) / denom;

  double Coef31 = b3 * a2 - b2 * a3;
  double Coef32 = a3 * b1 - a1 * b3;
  double Coef33 = b2 * a1 - b1 * a2;

  double G3 = (-v4(0) * Coef31 - v4(1) * Coef32 - v4(2) * Coef33) / denom;
  double H3 = (-p5(0) * Coef31 - p5(1) * Coef32 - p5(2) * Coef33) / denom;

  double C1 = 1 - K2;
  double C2 = 1;
  double C3 = -K2;
  double C4 = -2.0 * u1.dot(u2);
  double C5 = 2.0 * K2 * u1.dot(u3);
  double C6 = 2.0 * (u1.dot(p6) - K2 * u1.dot(p8));
  double C7 = -2.0 * u2.dot(p6);
  double C8 = 2.0 * K2 * u3.dot(p8);
  double C9 = p6.dot(p6) - K2 * p8.dot(p8);

  // Derive the coefficients of the quadratic
  double A =
      C1 * G1 * G1 + C2 * G2 * G2 + C3 * G3 * G3 + C4 * G1 * G2 + C5 * G1 * G3;
  double B = C1 * 2 * G1 * H1 + C2 * 2 * G2 * H2 + C3 * 2 * G3 * H3 +
             C4 * (G1 * H2 + G2 * H1) + C5 * (G1 * H3 + G3 * H1) + C6 * G1 +
             C7 * G2 + C8 * G3;
  double C = C1 * H1 * H1 + C2 * H2 * H2 + C3 * H3 * H3 + C4 * H1 * H2 +
             C5 * H1 * H3 + C6 * H1 + C7 * H2 + C8 * H3 + C9;
  double Disc = (B * B - 4 * A * C);

  Eigen::MatrixXd depths;
  depths.resize(4, 2);
  depths.setZero();

  if (Disc >= 0) {
    double sqrtD = sqrt(Disc);
    double s4a = (-B - sqrtD) / (2 * A);
    double s4b = (-B + sqrtD) / (2 * A);

    if (s4a > 0) {
      depths(0, 0) = G1 * s4a + H1;
      depths(1, 0) = G2 * s4a + H2;
      depths(2, 0) = G3 * s4a + H3;
      depths(3, 0) = s4a;
    }

    if (s4b > 0) {
      depths(0, 1) = G1 * s4b + H1;
      depths(1, 1) = G2 * s4b + H2;
      depths(2, 1) = G3 * s4b + H3;
      depths(3, 1) = s4b;
    }
  }

  return depths;
}

Eigen::MatrixXd BuildCoefficientMatrix(const Gp4pc::Input& input) {
  Eigen::MatrixXd coeff_mat(kNumConstraints, kNumMonomials);
  coeff_mat.setZero();

  // Useful aliases.
  // Camera centers.
  const std::vector<Eigen::Vector3d>& ray_origins = input.ray_origins;
  // Unit rays originating from ray_origins.
  const std::vector<Eigen::Vector3d>& ray_directions = input.ray_directions;
  // World points.
  const std::vector<Eigen::Vector3d>& world_points = input.world_points;

  // Verify the input is consistent.
  CHECK_EQ(ray_origins.size(), ray_directions.size());
  CHECK_EQ(ray_origins.size(), world_points.size());

  // The mathematical relationship is the following:
  //
  // ray_origins[i] + depth[i] * ray_directions[i] =
  //       s * R * world_points[i] + t,
  //
  // where s is scale, R is a rotation matrix, and t is translation.

  // Input Constants
  const Eigen::Vector3d& p1 = ray_origins[0];
  const Eigen::Vector3d& p2 = ray_origins[1];
  const Eigen::Vector3d& p3 = ray_origins[2];
  const Eigen::Vector3d& p4 = ray_origins[3];

  // Input Constants
  const Eigen::Vector3d& u1 = ray_directions[0];
  const Eigen::Vector3d& u2 = ray_directions[1];
  const Eigen::Vector3d& u3 = ray_directions[2];
  const Eigen::Vector3d& u4 = ray_directions[3];

  // Input Constants
  const Eigen::Vector3d& x1 = world_points[0];
  const Eigen::Vector3d& x2 = world_points[1];
  const Eigen::Vector3d& x3 = world_points[2];
  const Eigen::Vector3d& x4 = world_points[3];

  // Derived Constants
  double f1, f2, f3, f4, K1, K2;
  CalculateRatiosFromFourPoints(x1, x2, x3, x4, f1, f2, f3, f4, K1, K2);

  // Derived Constants
  const Eigen::Vector3d p5  = f1 * p1 + f2 * p2 - f3 * p3 - f4 * p4;
  const Eigen::Vector3d p6  = p1 - p2;
  const Eigen::Vector3d p7  = p3 - p4;
  const Eigen::Vector3d p8  = p1 - p3;
  const Eigen::Vector3d p9  = p2 - p4;
  const Eigen::Vector3d p10 = p1 - p4;
  const Eigen::Vector3d p11 = p2 - p3;

  // Derived Constants
  const Eigen::Vector3d v1 =  f1 * u1;
  const Eigen::Vector3d v2 =  f2 * u2;
  const Eigen::Vector3d v3 = -f3 * u3;
  const Eigen::Vector3d v4 = -f4 * u4;

  // Equation derived from orthogonality constraint 1.
  coeff_mat(0, 0)  =  v1.dot(u1);
  coeff_mat(0, 1)  = -u2.dot(v2);
  coeff_mat(0, 2)  =  0;
  coeff_mat(0, 3)  =  0;
  coeff_mat(0, 4)  =  u1.dot(v2) - u2.dot(v1);
  coeff_mat(0, 5)  =  u1.dot(v3);
  coeff_mat(0, 6)  =  u1.dot(v4);
  coeff_mat(0, 7)  = -u2.dot(v3);
  coeff_mat(0, 8)  = -u2.dot(v4);
  coeff_mat(0, 9)  =  0;
  coeff_mat(0,10)  =  p6.dot(v1) + u1.dot(p5);
  coeff_mat(0,11)  =  p6.dot(v2) - u2.dot(p5);
  coeff_mat(0,12)  =  p6.dot(v3);
  coeff_mat(0,13)  =  p6.dot(v4);
  coeff_mat(0,14)  =  p6.dot(p5);

  // Equation derived from orthogonality constraint 2.
  coeff_mat(1,0)  =  0;
  coeff_mat(1,1)  =  0;
  coeff_mat(1,2)  =  u3.dot(v3);
  coeff_mat(1,3)  = -u4.dot(v4);
  coeff_mat(1,4)  =  0;
  coeff_mat(1,5)  =  u3.dot(v1);
  coeff_mat(1,6)  = -u4.dot(v1);
  coeff_mat(1,7)  =  u3.dot(v2);
  coeff_mat(1,8)  = -u4.dot(v2);
  coeff_mat(1,9) =  u3.dot(v4) - u4.dot(v3);
  coeff_mat(1,10) =  p7.dot(v1);
  coeff_mat(1,11) =  p7.dot(v2);
  coeff_mat(1,12) =  p7.dot(v3) + u3.dot(p5);
  coeff_mat(1,13) =  p7.dot(v4) - u4.dot(p5);
  coeff_mat(1,14) =  p7.dot(p5);

  // Coefficients of quadratic polynomial that corresponds to the squared
  // length for edges 1--2, 1--3 and 3--4.
  Eigen::VectorXd d12(kNumMonomials);
  Eigen::VectorXd d13(kNumMonomials);
  Eigen::VectorXd d34(kNumMonomials);
  d12.setZero();
  d13.setZero();
  d34.setZero();

  d12(0)  =  1;
  d12(1)  =  1;
  d12(4)  = -2.0 * u1.dot(u2);
  d12(10) =  2.0 * u1.dot(p6);
  d12(11) = -2.0 * u2.dot(p6);
  d12(14) =  p6.dot(p6);

  d13(0)  =  1;
  d13(2)  =  1;
  d13(5)  = -2.0 * u1.dot(u3);
  d13(10) =  2.0 * u1.dot(p8);
  d13(12) = -2.0 * u3.dot(p8);
  d13(14) =  p8.dot(p8);

  d34(2)  =  1;
  d34(3)  =  1;
  d34(9)  = -2.0 * u3.dot(u4);
  d34(12) =  2.0 * u3.dot(p7);
  d34(13) = -2.0 * u4.dot(p7);
  d34(14) =  p7.dot(p7);

  // Two equations derived from distance ratio constraints.
  coeff_mat.row(2) = d12 - K1 * d34;
  coeff_mat.row(3) = d12 - K2 * d13;

  return coeff_mat;
}

bool IsSimilarityTransformValid(const Eigen::Matrix3d& rotation,
                                const Eigen::Vector3d& translation,
                                const double scale) {
  // Make sure rotation is finite.
  const double* rotation_vals = rotation.data();
  bool is_rotation_finite = true;
  for (int i = 0; i < 9; ++i) {
    is_rotation_finite = is_rotation_finite && std::isfinite(rotation_vals[i]);
  }
  // Make sure translation is finite.
  const bool is_translation_finite =
      std::isfinite(translation(0)) &&
      std::isfinite(translation(1)) &&
      std::isfinite(translation(2));
  // Make scale is finite.
  const bool is_scale_valid = scale > 0.0;
  if (is_rotation_finite && is_translation_finite && is_scale_valid) {
    return true;
  }

  return false;
}

void SolveForRotationAndTranslation(
    const Gp4pc::Input& input,
    const std::vector<Eigen::Vector4d>& plausible_depths,
    Gp4pc::Solution* solution) {
  solution->rotations.reserve(plausible_depths.size());
  solution->translations.reserve(plausible_depths.size());
  solution->scales.reserve(plausible_depths.size());
  solution->depths.reserve(plausible_depths.size());

  // Solve for similarity transform using Umeyama's method. See
  // theia/sfm/transformation/align_point_clouds.h for more information.
  std::vector<Eigen::Vector3d> camera_points(input.world_points.size());
  Eigen::Matrix3d rotation;
  Eigen::Vector3d translation;
  double scale;
  for (const Eigen::Vector4d& depth : plausible_depths) {
    // Compute the camera points.
    camera_points[0] = ComputeCameraPoint(
        input.ray_origins[0], input.ray_directions[0], depth[0]);
    camera_points[1] = ComputeCameraPoint(
        input.ray_origins[1], input.ray_directions[1], depth[1]);
    camera_points[2] = ComputeCameraPoint(
        input.ray_origins[2], input.ray_directions[2], depth[2]);
    camera_points[3] = ComputeCameraPoint(
        input.ray_origins[3], input.ray_directions[3], depth[3]);
    theia::AlignPointCloudsUmeyama(
        input.world_points, camera_points, &rotation, &translation, &scale);

    if (IsSimilarityTransformValid(rotation, translation, scale)) {
      solution->rotations.emplace_back(rotation);
      solution->translations.emplace_back(translation);
      solution->scales.emplace_back(scale);
      solution->depths.emplace_back(depth);
    }
  }
}

}  // namespace

Gp4pc::Gp4pc(const Params& params) : params_(params) {}

std::vector<Eigen::Vector4d>
Gp4pc::KeepPlausibleSolutions(const Eigen::MatrixXcd& solutions) {
  std::vector<Eigen::Vector4d> plausible_solutions;
  plausible_solutions.reserve(kMaxNumDepthSolutions);

  // Iterate through the possible solutions.
  Eigen::Vector4d estimated_solution;
  double discriminant = 0.0;
  double max_imag_entry = 0.0;
  Eigen::Vector4d prev_solution;
  double min_depth = 0.0;
  prev_solution.setZero();
  for (int i = 0; i < solutions.cols(); ++i) {
    max_imag_entry = std::max(std::abs(solutions(0, i).imag()),
                              std::abs(solutions(1, i).imag()));
    max_imag_entry = std::max(max_imag_entry, std::abs(solutions(2, i).imag()));
    max_imag_entry = std::max(max_imag_entry, std::abs(solutions(3, i).imag()));
    // TODO(vfragoso): Threshold solutions with large imaginary parts. Set a
    // a good threshold for getting good solutions with stmall imaginary parts.
    const bool real_solution = (max_imag_entry <= kRealNumberThreshold);
    VLOG(4) << "Max imag entry: " << max_imag_entry
            << " => real_solution: " << real_solution;

    // Keep the solutions that correspond to positive depths.
    estimated_solution[0] = solutions(0, i).real();
    estimated_solution[1] = solutions(1, i).real();
    estimated_solution[2] = solutions(2, i).real();
    estimated_solution[3] = solutions(3, i).real();

    min_depth = estimated_solution.minCoeff();
    const bool all_positive_depths = (min_depth >= 0.0);
    VLOG(4) << "Min. depth: " << min_depth
            << " => " << all_positive_depths
            << " " << estimated_solution.transpose();

    // Avoid duplicated solutions coming from those with small imaginary parts.
    // Remove duplicate solutions which are contigous.
    const double error = (prev_solution - estimated_solution).squaredNorm();
    const bool is_unique_soln = (error >= kDuplicateSolnErrorThreshold);

    // Save previous solution, regardless.
    VLOG(4) << "Unique: " << is_unique_soln << " "
            << prev_solution.transpose() << " <-> "
            << estimated_solution.transpose();
    prev_solution = estimated_solution;
    if (real_solution && all_positive_depths && is_unique_soln) {
      // Save solution.
      plausible_solutions.emplace_back(std::move(estimated_solution));
    }
  }

  VLOG(4) << "Plausible depths: " << plausible_solutions.size()
          << " out of " << solutions.cols();
  return plausible_solutions;
}

std::vector<Eigen::Vector4d>
Gp4pc::SolveForPoseViaPlanarSolver(const Gp4pc::Input& input) {
  VLOG(4) << "Solving pose using planar solver...";
  // Solve for depths via the planar solver.
  const Eigen::MatrixXcd depths = SolveForDepthsFromAPlanarInput(input);

  // Discard bad solutions.
  const std::vector<Eigen::Vector4d> final_depths =
      KeepPlausibleSolutions(depths);

  return final_depths;
}

std::vector<Eigen::Vector4d>
Gp4pc::SolveForPoseViaGeneralSolver(const Input& input) {
  VLOG(4) << "Solving for pose!";
  // Build coefficient matrix.
  const Eigen::MatrixXd coefficients_matrix = BuildCoefficientMatrix(input);
  VLOG(4) << "Coeff matrix: \n" << coefficients_matrix;

  // Solve for depths.
  const Eigen::MatrixXcd depths = SolveGp4pcPolynomial(coefficients_matrix);

  // Discard bad solutions.
  const std::vector<Eigen::Vector4d> final_depths =
      KeepPlausibleSolutions(depths);

  return final_depths;
}

bool Gp4pc::EstimateSimilarityTransformation(const Input& input,
                                             Solution* solution) {
  using Eigen::MatrixXcd;
  using Eigen::Vector4d;
  CHECK_EQ(input.ray_origins.size(), kSizeOfMinimalSample);
  CHECK_EQ(input.ray_origins.size(), input.ray_directions.size());
  CHECK_EQ(input.ray_origins.size(), input.world_points.size());
  CHECK_NOTNULL(solution)->rotations.clear();
  solution->translations.clear();
  solution->scales.clear();
  solution->depths.clear();

  // Validate input.
  if (!IsInputValid(input)) {
    return false;
  }

  // Each entry is the depth wrt to a camera center.
  MatrixXcd depths;
  std::vector<Vector4d> plausible_depths;
  plausible_depths.reserve(kMaxNumDepthSolutions);

  // Check whether the input is planar.
  bool is_planar = false;
  const bool use_planar_solver = !params_.use_general_solver;
  if (IsInputDegenerateOrPlanar(input,
                                params_.coplanar_threshold,
                                params_.colinear_threshold,
                                &is_planar)) {
    VLOG(4) << "Degenerate case. Could not compute pose.";
    return false;
  } else if (use_planar_solver && is_planar) {
    VLOG(4) << "Coplanar case.";
    // Solve for the planar case.
    plausible_depths = SolveForPoseViaPlanarSolver(input);
  } else {
    VLOG(4) << "Using general solver.";
    plausible_depths = SolveForPoseViaGeneralSolver(input);
  }

  if (plausible_depths.empty()) {
    return false;
  }
  VLOG(4) << "Number of plausible depths: " << plausible_depths.size();

  // Solve for rotation and translation.
  SolveForRotationAndTranslation(input, plausible_depths, solution);
  VLOG(4) << "Number of final solutions: " << solution->rotations.size();

  return !solution->rotations.empty();
}

}  // namespace msft
