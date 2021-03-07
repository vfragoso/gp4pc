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

#include <gp4pc/gp4pc_robust_estimator.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include <glog/logging.h>
#include <gp4pc/camera_feature_correspondence_2d_3d.h>
#include <gp4pc/util.h>

namespace msft {

// Minimal sample size.
constexpr int kMinimalSampleSize = 4;

Gp4pcRobustEstimator::Gp4pcRobustEstimator(const RansacParameters& params)
    : params_(params), prng_(params.seed) {
  // Make sure parameters are valid.
  CHECK_GT(params_.failure_probability, 0.0);
  CHECK_LT(params_.failure_probability, 1.0);
  CHECK_GT(params_.reprojection_error_thresh, 0.0);
  CHECK_GE(params_.min_iterations, 0);
  CHECK_GT(params_.max_iterations, params_.min_iterations);
}

int Gp4pcRobustEstimator::RandInt(const int min_value, const int max_value) {
  std::uniform_int_distribution<int> index_dist(min_value, max_value);
  return index_dist(prng_);
}

std::vector<CameraFeatureCorrespondence2D3D>
Gp4pcRobustEstimator::Sample(
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences) {
  std::vector<CameraFeatureCorrespondence2D3D> sample;
  sample.reserve(kMinimalSampleSize);
  const int num_correspondences = correspondences.size();
  for (int i = 0; i < kMinimalSampleSize; ++i) {
    // Randomly sample from the correspondence indices.
    std::swap(correspondence_indices_[i],
              correspondence_indices_[RandInt(i, num_correspondences - 1)]);
    // Copy correspondence.
    sample.push_back(correspondences[correspondence_indices_[i]]);
  }
  return sample;
}

double Gp4pcRobustEstimator::UpdateBestSolution(
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences,
    const Gp4pc::Solution& estimated_solns,
    Gp4pc::Solution* best_solution,
    std::vector<int>* best_inliers) {
  const int num_estimated_solns = estimated_solns.rotations.size();
  // Check reprojection errors for every point.
  const double sq_reprojection_error_thresh =
      params_.reprojection_error_thresh * params_.reprojection_error_thresh;

  std::vector<int> inliers;
  inliers.reserve(correspondences.size());
  Eigen::Vector2d pixel;
  double best_inlier_ratio =
      best_inliers->size() / static_cast<double>(correspondences.size()) +
      std::numeric_limits<double>::epsilon();
  for (int i = 0; i < num_estimated_solns; ++i) {
    const Eigen::Quaterniond& rotation = estimated_solns.rotations[i];
    const Eigen::Vector3d& translation = estimated_solns.translations[i];
    const double& scale = estimated_solns.scales[i];
    inliers.clear();
    VLOG(3) << "Rotation matrix: \n" << rotation.toRotationMatrix();
    VLOG(3) << "Translation: " << translation.transpose();
    VLOG(3) << "Scale: " << scale;
    for (int j = 0; j < correspondences.size(); ++j) {
      // Compute point coordinates wrt generalized coordinate frame:
      //   cam_position + depth * ray = scale * rotation * point + translation
      //   depth * ray = scale * rotation * point + translation - cam_position.
      const Eigen::Vector3d point_in_gen_camera =
          scale * (rotation * correspondences[j].point) + translation;
      // Project point in camera.
      const PinholeCamera& camera = correspondences[j].camera;
      if (camera.ProjectPoint(point_in_gen_camera, &pixel) < 0) {
        continue;
      }
      // Reprojection error.
      const double sq_reprojection_error =
          (pixel - correspondences[j].observation).squaredNorm();
      // Is it an inlier?
      if (sq_reprojection_error < sq_reprojection_error_thresh) {
        inliers.push_back(j);
      }
    }
    // Do we have more inliers than the best solution? If so, then update best
    // solution and inliers.
    if (best_inliers->size() < inliers.size()) {
      *best_inliers = inliers;
      best_solution->rotations[0] = rotation;
      best_solution->translations[0] = translation;
      best_solution->scales[0] = scale;
      best_inlier_ratio =
          best_inliers->size() / static_cast<double>(correspondences.size());
      VLOG(3) << "Update num. inliers: " << best_inliers->size();
      VLOG(3) << "Update inlier ratio: " << best_inlier_ratio;
    }
  }

  return best_inlier_ratio;
}

int Gp4pcRobustEstimator::ComputeMaxIterations(
    const double inlier_ratio,
    const double log_failure_prob) {
  CHECK_GT(inlier_ratio, 0.0);
  if (inlier_ratio == 1.0) {
    return params_.min_iterations;
  }

  // Log. probability of producing a bad hypothesis.
  const double log_prob =
      std::log(1.0 - pow(inlier_ratio, kMinimalSampleSize)) -
      std::numeric_limits<double>::epsilon();

  // Compute the number of iterations to achieve a certain confidence.
  const int num_iterations = static_cast<int>(log_failure_prob / log_prob);

  return std::clamp(num_iterations,
                    params_.min_iterations,
                    params_.max_iterations);
}

Gp4pc::Solution Gp4pcRobustEstimator::Estimate(
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences,
    RansacSummary* ransac_summary) {
  RansacSummary& summary = *CHECK_NOTNULL(ransac_summary);
  summary.inliers.clear();

  // Check that we have enough correspondences to produce a single hypothesis.
  CHECK(correspondences.size() >= kMinimalSampleSize)
      << "Not enough correspondences.";

  // Initialize correspondence indices.
  correspondence_indices_.clear();
  correspondence_indices_.resize(correspondences.size());
  std::iota(correspondence_indices_.begin(), correspondence_indices_.end(), 0);

  // The hypothesis-and-test loop.
  const double log_failure_prob = std::log(params_.failure_probability);
  int max_iterations = params_.max_iterations;
  std::vector<CameraFeatureCorrespondence2D3D> sample;
  Gp4pc::Solution hypotheses;
  // Initialize best solution to identity solution.
  Gp4pc::Solution best_solution;
  best_solution.rotations.push_back(Eigen::Quaterniond::Identity());
  best_solution.translations.push_back(Eigen::Vector3d::Zero());
  best_solution.scales.push_back(1.0);
  Gp4pc::Input input;
  double inlier_ratio = 0.0;
  for (summary.num_iterations = 0;
       summary.num_iterations < max_iterations;
       ++summary.num_iterations) {
    // Compute a minimal sample to produce hypotheses.
    sample = Sample(correspondences);

    // Compute hypotheses.
    input = ComputeInputDatum(sample);

    if (!estimator_.EstimateSimilarityTransformation(input, &hypotheses)) {
      VLOG(3) << "Failed to estimate hypotheses. Skipping sample ...";
      continue;
    }

    summary.num_hypotheses += hypotheses.rotations.size();
    VLOG(3) << "Num. candidate solutions: " << hypotheses.rotations.size();

    // Update best solution.
    inlier_ratio = UpdateBestSolution(correspondences,
                                      hypotheses,
                                      &best_solution,
                                      &summary.inliers);

    // Update max. iterations.
    max_iterations = ComputeMaxIterations(inlier_ratio, log_failure_prob);
  }

  // Compute confidence.
  summary.confidence =
      1.0 - std::pow(1.0 - std::pow(inlier_ratio, kMinimalSampleSize),
                     summary.num_iterations);
  VLOG(3) << "Best inlier ratio: " << inlier_ratio;
  VLOG(3) << "Confidence: " << summary.confidence;
  return best_solution;
}

}  // namespace msft
