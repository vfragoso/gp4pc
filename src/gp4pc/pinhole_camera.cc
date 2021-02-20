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

#include <gp4pc/pinhole_camera.h>

#include <cmath>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>

namespace msft {

PinholeCamera::PinholeCamera(const double focal_length,
                             const Eigen::Vector2d& principal_point,
                             const Eigen::Quaterniond& world_to_camera_rotation,
                             const Eigen::Vector3d& world_to_camera_translation)
    : rotation_(world_to_camera_rotation.normalized()),
      translation_(world_to_camera_translation) {
  intrinsics_mat_ <<
      focal_length, 0.0, principal_point.x(),
      0.0, focal_length, principal_point.y(),
      0.0, 0.0, 1.0;
}

PinholeCamera::PinholeCamera(const double focal_length,
                             const Eigen::Vector2d& principal_point,
                             const Eigen::Vector4d& world_to_camera_rotation,
                             const Eigen::Vector3d& world_to_camera_translation)
    : PinholeCamera(focal_length,
                    principal_point,
                    Eigen::Quaterniond(world_to_camera_rotation[0],
                                       world_to_camera_rotation[1],
                                       world_to_camera_rotation[2],
                                       world_to_camera_rotation[3]),
                    world_to_camera_translation) {
  CHECK(std::abs(world_to_camera_rotation.norm() - 1.0) <
        std::numeric_limits<double>::epsilon())
      << "Quaternion has not unit norm";
}

double PinholeCamera::ProjectPoint(const Eigen::Vector3d& point3d,
                                   Eigen::Vector2d* point2d) const {
  const Eigen::Vector3d point_in_camera = rotation_ * point3d + translation_;
  *CHECK_NOTNULL(point2d) = (intrinsics_mat_ * point_in_camera).hnormalized();
  return point_in_camera.z();
}

Eigen::Vector3d
PinholeCamera::PixelToUnitRay(const Eigen::Vector2d& pixel) const {
  const Eigen::Vector3d ray =
      rotation_.conjugate() *
      (intrinsics_mat_.inverse() * pixel.homogeneous()).normalized();
  return ray;
}

Eigen::Vector3d PinholeCamera::GetPosition() const {
  const Eigen::Vector3d camera_position = rotation_.conjugate() * -translation_;
  return camera_position;
}

}  // namespace msft
