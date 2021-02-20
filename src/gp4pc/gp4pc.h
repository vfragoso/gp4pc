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

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace msft {

// TODO(vfragoso): Document class!
class Gp4pc {
public:
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
  // computes. The structure include the rotations, translations, and scales.
  struct Solution {
    std::vector<Eigen::Quaterniond> rotations;
    std::vector<Eigen::Vector3d> translations;
    std::vector<double> scales;
  };

  // Default constructor and destructor.
  Gp4pc() = default;
  ~Gp4pc() = default;

  // Estimates the similarity transformations from the given 2D-3D
  // correspondences and priors.
  //
  // Params:
  //   input  The 2D-3D correspondences and priors.
  //   solution  The structure holding all the solutions found.
  bool EstimateSimilarityTransformation(const Input& input, Solution* solution);
};

}  // namespace msft

#endif  // GP4PC_GP4PC_H_