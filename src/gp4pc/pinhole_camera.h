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

#ifndef GP4P_PINHOLE_CAMERA_
#define GP4P_PINHOLE_CAMERA_

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace msft {

// Simple pinhole camera class which is useful for estimating similarity
// transformation from 2D and 3D correspondences. It is useful for a robust
// estimation that aims to minimize the reprojection error.
// This class uses the following projection model:
//
//  | u |           | focal_length      0        principal_point.x |   | x |
//  | v | = 1 / z * |      0       focal_length  principal_point.y | * | y |,
//  | 1 |           |      0            0                 1        |   | z |
//
// where [x, y, z]^T is the 3D point position wrt to the camera frame, and
// [u, v, 1]^T is the 2D pixel projection. To obtain [x, y, z]^T, we need to
// map the 3D point wrt to the world coordinate frame using the extrinsics of
// the camera:
//
//  | x |  
//  | y | = [R | t] * p, 
//  | z |
//
// where R and t are the rotation and translation that map a point from the
// world to the camera coordinate system.
class PinholeCamera {
 public:
  // Params:
  //   focal_length  The focal length of the camera.
  //   principal_point  The principal point of the camera.
  //   world_to_camera_rotation  The unit-quaternion encoding the rotation
  //      that maps from world to camera coordinate systems.
  //   world_to_camera_translation  The translation vector from world to camera.
  PinholeCamera(const double focal_length,
                const Eigen::Vector2d& principal_point,
                const Eigen::Quaterniond& world_to_camera_rotation,
                const Eigen::Vector3d& world_to_camera_translation);

  // This constructor was designed for the Python integration.
  // Params:
  //   focal_length  The focal length of the camera.
  //   principal_point  The principal point of the camera.
  //   world_to_camera_rotation  The unit-quaternion encoding the rotation
  //      that maps from world to camera coordinate systems. The vector must
  //      follow the next format:
  //         w = world_to_camera_rotation[0]
  //         x = world_to_camera_rotation[1]
  //         y = world_to_camera_rotation[2]
  //         z = world_to_camera_rotation[3].
  //   world_to_camera_translation  The translation vector from world to camera.
  PinholeCamera(const double focal_length,
                const Eigen::Vector2d& principal_point,
                const Eigen::Vector4d& world_to_camera_rotation,
                const Eigen::Vector3d& world_to_camera_translation);  

  // Default constructor.
  PinholeCamera() : PinholeCamera(
      1.0,
      Eigen::Vector2d::Zero(),
      Eigen::Quaterniond::Identity(),
      Eigen::Vector3d::Zero()) {}

  // Destructor.
  ~PinholeCamera() = default;

  // Project 3D point using the focal_length and principal point parameters.
  // The function returns the z component of the point in front of camera. This
  // useful to know if it is in-front or behind the camera.
  double ProjectPoint(const Eigen::Vector3d& point3d,
                      Eigen::Vector2d* point2d) const;

  // Returns a unit-vector with direction to a 3D point that starts from the
  // center of the camera. The returned ray vector is described wrt to the world
  // coordinate system. This is so that
  //
  //   X = c + r * d
  //
  // is satisfied. Here X is a 3D point wrt to the world, c is the camera origin
  // wrt to the world, r is the unit-vector (or ray), and d is the depth of the
  // 3D point with respect to the camera origin
  Eigen::Vector3d PixelToUnitRay(const Eigen::Vector2d& pixel) const;

  // Returns the camera position wrt to the world. This is obtained from the
  // extrinsics parameters:
  //
  //   c = R^-1 * t.
  Eigen::Vector3d GetPosition() const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
 private:
  // Projection matrix.
  Eigen::Matrix3d intrinsics_mat_;
  // Extrinsics - World to camera rigid transformation.
  Eigen::Quaterniond rotation_;
  Eigen::Vector3d translation_;
};

}  // msft

#endif  // GP4P_PINHOLE_CAMERA_
