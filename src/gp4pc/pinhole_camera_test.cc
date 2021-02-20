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

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gp4pc/pinhole_camera.h>
#include <math/utils.h>

namespace msft {
namespace {

// Number of trials.
constexpr int kNumTrials = 32;
// Fixed focal length.
constexpr double kFocalLength = 10;

TEST(PinholeCameraTest, BasicPointProjection) {
  const Eigen::Vector2d principal_point(256, 256);
  const Eigen::Quaterniond rotation(
      Eigen::AngleAxisd(DegToRad(35.0),
                        Eigen::Vector3d::Random().normalized()));
  const Eigen::Vector3d translation = Eigen::Vector3d::Random();
  const PinholeCamera camera(kFocalLength,
                             principal_point,
                             rotation,
                             translation);
  const Eigen::Vector3d camera_position = rotation.conjugate() * -translation;
  Eigen::Vector2d pixel;
  for (int i = 0; i < kNumTrials; ++i) {
    const Eigen::Vector3d point_in_camera = Eigen::Vector3d::Random();
    const Eigen::Vector3d point_in_world =
        rotation.conjugate() * point_in_camera + camera_position;
    const double u =
        (kFocalLength * point_in_camera.x() / point_in_camera.z()) +
        principal_point.x();
    const double v =
        (kFocalLength * point_in_camera.y() / point_in_camera.z()) +
        principal_point.y();
    const Eigen::Vector2d expected_pixel(u, v);
    EXPECT_NEAR(camera.ProjectPoint(point_in_world, &pixel),
                point_in_camera.z(),
                1e-6);
    EXPECT_NEAR((expected_pixel - pixel).squaredNorm(), 0.0, 1e-6);
  }
}

TEST(PinholeCameraTest, BasicPixelToUnitRay) {
  const Eigen::Vector2d principal_point(256, 256);
  const Eigen::Quaterniond rotation(
      Eigen::AngleAxisd(DegToRad(35.0),
                        Eigen::Vector3d::Random().normalized()));
  const Eigen::Vector3d translation = Eigen::Vector3d::Random();
  const PinholeCamera camera(kFocalLength,
                             principal_point,
                             rotation,
                             translation);
  const Eigen::Vector3d camera_position = camera.GetPosition();
  for (int i = 0; i < kNumTrials; ++i) {
    // Point in front of camera.
    Eigen::Vector3d point_in_camera = Eigen::Vector3d::Random();
    point_in_camera.z() = std::abs(point_in_camera.z());
    const Eigen::Vector3d point_in_world =
        rotation.conjugate() * point_in_camera + camera_position;
    const Eigen::Vector3d unnormalized_ray = point_in_world - camera_position;
    const Eigen::Vector3d expected_ray = unnormalized_ray.normalized();
    Eigen::Vector2d pixel;
    camera.ProjectPoint(point_in_world, &pixel);
    const Eigen::Vector3d ray = camera.PixelToUnitRay(pixel);
    EXPECT_NEAR((ray - expected_ray).squaredNorm(), 0.0, 1e-6);
  }
}

TEST(PinholeCameraTest, CameraPosition) {
  const Eigen::Vector2d principal_point(256, 256);
  for (int i = 0; i < kNumTrials; ++i) {
    const Eigen::Quaterniond rotation(
        Eigen::AngleAxisd(DegToRad(35.0),
                          Eigen::Vector3d::Random().normalized()));
    const Eigen::Vector3d translation = Eigen::Vector3d::Random();
    const PinholeCamera camera(kFocalLength,
                               principal_point,
                               rotation,
                               translation);
    const Eigen::Vector3d expected_position =
        rotation.conjugate() * -translation;
    EXPECT_NEAR((expected_position - camera.GetPosition()).squaredNorm(),
                0.0,
                1e-6);
  }
}

}  // namespace
}  // namespace msft
