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

#ifndef GP4PC_GP4PC_ROBUST_ESTIMATOR_H_
#define GP4PC_GP4PC_ROBUST_ESTIMATOR_H_

#include <vector>
#include <random>
#include <gp4pc/gp4pc.h>

namespace msft {
// Forward declaration.
struct CameraFeatureCorrespondence2D3D;

class Gp4pcRobustEstimator {
 public:
  struct RansacParameters {
    // The failure probability of RANSAC. This is useful for estimating the
    // number of iterations in RANSAC adaptively. Setting this probability
    // to 0.01 is equivalent to expect that there is a 1% chance of missing
    // the correct estimate given a minimal sample.
    double failure_probability = 0.01;

    // Reprojection error threshold (in pixels).
    double reprojection_error_thresh = 2.0;

    // Minimum number of iterations.
    int min_iterations = 100;

    // Maximum number of iterations.
    int max_iterations = 1000;

    // Random seed.
    size_t seed = 67;
  };

  // This structure contains statistics about the RANSAC run as well as
  // the set of found inliers.
  struct RansacSummary {
    // Inlier indices.
    std::vector<int> inliers;

    // Number of iterations performed in the RANSAC estimation process.
    int num_iterations = 0;

    // The confidence in the solution.
    double confidence = 0.0;

    // Number of evaluated hypotheses.
    int num_hypotheses = 0;
  };

  explicit Gp4pcRobustEstimator(const RansacParameters& params);
  ~Gp4pcRobustEstimator() = default;

  // Estimates the similarity transformation using gDLS* as a minimal solver.
  Gp4pc::Solution Estimate(
      const std::vector<CameraFeatureCorrespondence2D3D>& correspondences,
      RansacSummary* ransac_summary);

 private:
  // Ransac parameters.
  const RansacParameters params_;

  // Pseudo random number generator.
  std::mt19937 prng_;

  // Gp4pc estimator.
  Gp4pc estimator_;

  // Correspondence indices.
  std::vector<int> correspondence_indices_;

  // Helper functions.
  // Computes a random minimal sample. This is used to generate hypotheses.
  std::vector<CameraFeatureCorrespondence2D3D> Sample(
      const std::vector<CameraFeatureCorrespondence2D3D>& correspondences);

  // Regnerates random integer within a specific range.
  int RandInt(const int min_value, const int max_value);

  // Computes maximum number of iterations as a function of inlier ratio and
  // probability of failure. This functions operates as follows: given the
  // current inlier_ratio, and the probability of failure, the number of
  // iterations that are required to find a good hypothesis is
  //
  //  num_iterations = log_failure_prob / log(1 - inlier_ratio^min_sample),
  //
  // where min_sample is the minimum sample size to produce a hypothesis.
  // For more information, please see
  // https://en.wikipedia.org/wiki/Random_sample_consensus.
  int ComputeMaxIterations(const double inlier_ratio,
                           const double log_failure_prob);

  // Updates the best solution by identifying inliers and keeping the solution
  // that has the largest number of inliers.
  double UpdateBestSolution(
      const std::vector<CameraFeatureCorrespondence2D3D>& correspondences,
      const Gp4pc::Solution& estimated_solns,
      Gp4pc::Solution* best_solution,
      std::vector<int>* best_inlier_idxs);
};

}  // namespace msft

#endif  // GP4PC_GP4PC_ROBUST_ESTIMATOR_H_