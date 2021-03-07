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

#ifndef GP4PC_GP4PC_H_
#define GP4PC_GP4PC_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace msft {

// This class implements a minimal solver for the generalized pose-and-scale
// problem descrobed in
//
// Victor Fragoso and Sudipta Sinha.
// In Proc. of the IEEE International Conf. on 3D Vision (3DV) 2020.
//
// The main idea is to exploit 4-point congruence constraints that can be
// obtained from ratios relating the lengths of a tethrahedron formed by the
// 2D-3D correspondences. These constraints help us formulate a problem that
// depends only on the depths (distance between center of projection and the
// 3D point). Thus, GP4PC solves first for the depths, and the solves a 3D-3D
// alignment problem via Umeyama's method.
//
// We use the Umeyama's method implemented in Theia. This method assumes the
// following relationship:
//
//   camera_center + depth * ray = scale * rotation * point + translation.
class Gp4pc {
 public:
  // Parameters determining behavior of the planar solver and whether to use
  // only the general solver or hybrad version.
  struct Params {
    // Coplanar threshold.
    double coplanar_threshold = 1e-3;
    // Colinear threshold.
    double colinear_threshold = 1e-2;
    // Whether to always use the general solver, or a hybrid (planar + general)
    // otherwise. When set to true, the general solver is the default one.
    // Otherwise, the hybrid version (planar + general solvers) kicks in.
    bool use_general_solver = false;
  };

  // This structure aims to collect the input for gp4pc. The input mainly aims
  // to encode the 2D-3D correspondences and the input includes:
  //   1. Ray origins (camera positions).
  //   2. Ray directions (direction vector from ray origin to a 3D point).
  //   3. World point (3D point in the world).
  // Note that ray origin and direction are referenced wrt the generalized
  // camera coordinate system.
  struct Input {
    std::vector<Eigen::Vector3d> ray_origins;
    std::vector<Eigen::Vector3d> ray_directions;
    std::vector<Eigen::Vector3d> world_points;
  };

  // This structure encodes all the similarity transformations that gp4pc
  // computes. The structure include the rotations, translations, scales,
  // and depths. The solution thus satisfies
  //
  //   camera_center + depth * ray = scale * rotation * point + translation.
  struct Solution {
    std::vector<Eigen::Quaterniond> rotations;
    std::vector<Eigen::Vector3d> translations;
    std::vector<double> scales;
    std::vector<Eigen::Vector4d> depths;
  };

  // Constructor.
  explicit Gp4pc(const Params& params);
  // Default constructor and destructor.
  Gp4pc() : Gp4pc(Params()) {}
  ~Gp4pc() = default;

  // Estimates the similarity transformations from the given 2D-3D
  // correspondences and priors.
  //
  // Params:
  //   input  The 2D-3D correspondences and priors.
  //   solution  The structure holding all the solutions found.
  bool EstimateSimilarityTransformation(const Input& input, Solution* solution);

 private:
  // Parameters.
  const Params params_;

  std::vector<Eigen::Vector4d> SolveForPoseViaPlanarSolver(const Input& input);

  std::vector<Eigen::Vector4d> SolveForPoseViaGeneralSolver(const Input& input);

  std::vector<Eigen::Vector4d>
  KeepPlausibleSolutions(const Eigen::MatrixXcd& solutions);
};

}  // namespace msft

#endif  // GP4PC_GP4PC_H_
