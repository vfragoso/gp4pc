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

#include <gp4pc/estimate_similarity_transformation.h>

#include <vector>
#include <gp4pc/camera_feature_correspondence_2d_3d.h>
#include <gp4pc/gp4pc.h>
#include <gp4pc/gp4pc_robust_estimator.h>
#include <gp4pc/util.h>

namespace msft {

// Computes the similarity transformation given the 2D-3D correspondences.
Gp4pc::Solution EstimateSimilarityTransformation(
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences) {
  Gp4pc::Solution solution;
  const Gp4pc::Input input = ComputeInputDatum(correspondences);
  Gp4pc estimator;
  estimator.EstimateSimilarityTransformation(input, &solution);  
  return solution;
}

// Computes the similarity transformation given 2D-3D correspondences using a
// RANSAC estimator.
Gp4pc::Solution EstimateSimilarityTransformation(
    const Gp4pcRobustEstimator::RansacParameters& params,
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences,
    Gp4pcRobustEstimator::RansacSummary* ransac_summary) {
  Gp4pcRobustEstimator estimator(params);
  const Gp4pc::Solution solution =
      estimator.Estimate(correspondences, ransac_summary);
  return solution;
}

}  // namespace msft
