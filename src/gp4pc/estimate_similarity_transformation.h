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

#ifndef GP4PC_ESTIMATE_SIMILARITY_TRANSFORMATION_H_
#define GP4PC_ESTIMATE_SIMILARITY_TRANSFORMATION_H_

#include <Eigen/Core>
#include <gp4pc/camera_feature_correspondence_2d_3d.h>
#include <gp4pc/gp4pc.h>
#include <gp4pc/gp4pc_robust_estimator.h>

namespace msft {

// Computes the similarity transformation given 2D-3D correspondences.
// The 2D-3D correspondences contains also the camera that observes the
// projection of the candidate 3D point. This function calls the gDLS*
// directly. Thus, this function can use the gDLS* estimator as minimal
// or non-minimal solver.
//
// Params:
//   correspondences  The 2D-3D correspondences.
Gp4pc::Solution EstimateSimilarityTransformation(
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences);

// Computes the similarity transformation using a RANSAC estimation framework.
// The function expects the ransac parameters, priors, 2D-3D correspondences,
// and the ransac_summary pointer to store RANSAC statistics. The function
// returns the estimated solution with the largest number of inliers.
//
// Params:
//   ransac_params  Ransac parameters.
//   correspondences  The 2D-3D correspondences.
//   ransac_summary  Ransac summary.
Gp4pc::Solution EstimateSimilarityTransformation(
    const Gp4pcRobustEstimator::RansacParameters& ransac_params,
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences,
    Gp4pcRobustEstimator::RansacSummary* ransac_summary);

}  // namespace msft

#endif  // GP4PC_ESTIMATE_SIMILARITY_TRANSFORMATION_H_
