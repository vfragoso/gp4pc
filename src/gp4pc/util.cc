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

#include <gp4pc/util.h>

#include <vector>
#include <Eigen/Core>
#include <gp4pc/camera_feature_correspondence_2d_3d.h>
#include <gp4pc/gp4pc.h>

namespace msft {

Gp4pc::Input ComputeInputDatum(
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences) {
  Gp4pc::Input input;
  input.ray_origins.resize(correspondences.size());
  input.ray_directions.resize(correspondences.size());
  input.world_points.resize(correspondences.size());
  for (int i = 0; i < correspondences.size(); ++i) {
    // Keep 3D point.
    input.world_points[i] = correspondences[i].point;
    // Compute ray direction.
    const Eigen::Vector2d& pixel = correspondences[i].observation;
    input.ray_directions[i] = correspondences[i].camera.PixelToUnitRay(pixel);
    // Compute ray origins.
    input.ray_origins[i] = correspondences[i].camera.GetPosition();
  }
  return input;
}

}  // namespace msft
