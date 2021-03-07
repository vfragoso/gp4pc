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

#ifndef GP4PC_CAMERA_FEATURE_CORRESPONDENCE_2D_3D_H_
#define GP4PC_CAMERA_FEATURE_CORRESPONDENCE_2D_3D_H_

#include <Eigen/Core>
#include <gp4pc/pinhole_camera.h>

namespace msft {

// The goal of this structure is to hold the 2D-3D correspondence as well as
// the individual camera observing the 3D point. This is necessary to build
// the input to gp4pc as it uses a generalized camera model.
struct CameraFeatureCorrespondence2D3D {
  // The pinhole camera seeing 2D feature. The camera pose must be wrt to the
  // generalized camera coordinate system (or query coordinate system).
  PinholeCamera camera;

  // Observed keypoint or 2D feature.
  Eigen::Vector2d observation;

  // Corresponding 3D point. This point is described wrt to the world coordinate
  // system.
  Eigen::Vector3d point;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

}  // namespace msft

#endif  // GP4PC_CAMERA_FEATURE_CORRESPONDENCE_2D_3D_H_
