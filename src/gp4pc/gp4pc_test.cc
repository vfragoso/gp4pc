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

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <gp4pc/gp4pc.h>
#include <gtest/gtest.h>
#include <math/utils.h>
#include <random>

namespace msft {
namespace {

// Random seed.
const int kSeed = 67;

using Eigen::AngleAxisd;
using Eigen::Quaterniond;
using Eigen::Vector3d;

// Useful aliases.
using Input = Gp4pc::Input;
using Solution = Gp4pc::Solution;

struct TestParams {
  double max_reprojection_error;
  double max_rotation_difference;
  double max_translation_difference;
  double max_scale_difference;
  double projection_noise_std_dev;
  double max_depth_sq_error;
  double max_alignment_sq_error;
  Gp4pc::Params solver_params;
};

struct ExpectedSolution {
  Quaterniond rotation;
  Vector3d translation;
  double scale;
};

struct TestInput {
  std::vector<Vector3d> camera_centers;
  std::vector<Vector3d> world_points;
};

// Fixture to test the solver.
class Gp4pcTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    // Create random number generator.
    rng = new std::mt19937(kSeed);
  }

  static void TearDownTestCase() {
    delete rng;
  }

  static Input GenerateTestingData(const ExpectedSolution& expected_solution,
                                   const TestParams& test_params,
                                   const TestInput& test_input,
                                   Eigen::Vector4d* depths);

  static void TestWithNoise(const ExpectedSolution& expected_solution,
                            const TestParams& test_params,
                            const TestInput& test_input,
                            const bool use_planar_solver);

  static void TestWithNoise(const ExpectedSolution& expected_solution,
                            const TestParams& test_params,
                            const TestInput& test_input);

  static void AddNoiseToRay(const double nose_std_dev, Eigen::Vector3d* ray);

  static std::mt19937* rng;
};

std::mt19937* Gp4pcTest::rng = nullptr;

inline bool CheckDepthErrors(const Eigen::Vector4d& estimated_depths,
                             const Eigen::Vector4d& gt_depths,
                             const double max_depth_sq_error) {
  const double sq_error = (estimated_depths - gt_depths).squaredNorm();
  return sq_error < max_depth_sq_error;
}

bool CheckReprojectionErrors(const Input& input,
                             const Eigen::Quaterniond& soln_rotation,
                             const Eigen::Vector3d& soln_translation,
                             const double soln_scale,
                             const double max_reprojection_error) {
  const int num_points = input.world_points.size();
  const std::vector<Vector3d>& camera_rays = input.ray_directions;
  const std::vector<Vector3d>& ray_origins = input.ray_origins;
  const std::vector<Vector3d>& world_points = input.world_points;
  double good_reprojection_errors = true;
  for (int i = 0; i < num_points; ++i) {
    const Quaterniond unrot =
        Quaterniond::FromTwoVectors(camera_rays[i], Vector3d(0, 0, 1));
    const Vector3d reprojected_point =
        soln_scale * (soln_rotation * world_points[i]) + soln_translation -
        ray_origins[i];

    const Vector3d unrot_cam_ray = unrot * camera_rays[i];
    const Vector3d unrot_reproj_pt = unrot * reprojected_point;
    
    const double reprojection_error = 
        (unrot_cam_ray.hnormalized() - unrot_reproj_pt.hnormalized()).norm();
    good_reprojection_errors = (good_reprojection_errors &&
                                (reprojection_error < max_reprojection_error));
  }
  return good_reprojection_errors;
}

void Gp4pcTest::AddNoiseToRay(const double noise_std_dev,
                              Eigen::Vector3d* ray) {
  static std::normal_distribution noise_dist(0.0, noise_std_dev);
  const double scale = CHECK_NOTNULL(ray)->norm();
  const double noise_x = noise_dist(*rng);
  const double noise_y = noise_dist(*rng);

  Eigen::Quaterniond rot =
      Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), *ray);
  Eigen::Vector3d noisy_point(noise_x, noise_y, 1);
  noisy_point *= scale;

  *ray = (rot * noisy_point).normalized();
}

Input Gp4pcTest::GenerateTestingData(const ExpectedSolution& expected_solution,
                                     const TestParams& test_params,
                                     const TestInput& test_input,
                                     Eigen::Vector4d* depths) {
  Input input;
  const int num_points = test_input.world_points.size();
  CHECK_EQ(num_points, 4) << "Num. of world points must be 4.";
  const int num_cameras = test_input.camera_centers.size();
  std::vector<Vector3d>& camera_rays = input.ray_directions;
  std::vector<Vector3d>& ray_origins = input.ray_origins;
  camera_rays.reserve(num_points);
  ray_origins.reserve(num_points);
  input.world_points = test_input.world_points;
  const std::vector<Vector3d>& world_points = test_input.world_points;
  const std::vector<Vector3d>& camera_centers = test_input.camera_centers;
  const Quaterniond expected_rotation = expected_solution.rotation;
  const Vector3d expected_translation = expected_solution.translation;
  const double expected_scale = expected_solution.scale;
  Eigen::Vector3d ray;
  for (int i = 0; i < num_points; ++i) {
    // Setting camera positions or ray origins.
    ray_origins.emplace_back(
        expected_scale * (expected_rotation * camera_centers[i % num_cameras]) +
        expected_translation);

    // Reproject 3D points into camera frame.
    ray = (expected_scale * (expected_rotation * world_points[i]) +
           expected_translation -
           ray_origins[i]);
    (*depths)(i) = ray.norm();
    camera_rays.emplace_back(ray.normalized());
  }

  // Add noise to the camera rays.
  if (test_params.projection_noise_std_dev > 0.0) {
    // Adds noise to both of the rays.
    for (int i = 0; i < num_points; ++i) {
      AddNoiseToRay(test_params.projection_noise_std_dev, &camera_rays[i]);
    }
  }

  return input;
}

void Gp4pcTest::TestWithNoise(const ExpectedSolution& expected_solution,
                              const TestParams& test_params,
                              const TestInput& test_input,
                              const bool use_planar_solver) {
  static const double kDefaultCoplanarThresh = 1e-3;
  static const double kDefaultColinearThresh = 1e-2;
  // Generate testing data.
  Eigen::Vector4d depths;
  const Input input =
      GenerateTestingData(expected_solution, test_params, test_input, &depths);

  // Estimate.
  Solution solution;
  Gp4pc solver(test_params.solver_params);
  EXPECT_TRUE(solver.EstimateSimilarityTransformation(input, &solution));

  // Check the solutions here.
  bool matched_transform = false;
  EXPECT_GT(solution.rotations.size(), 0);
  for (int i = 0; i < solution.rotations.size(); ++i) {
    const bool good_reprojection_errors =
        CheckReprojectionErrors(input,
                                solution.rotations[i],
                                solution.translations[i],
                                solution.scales[i],
                                test_params.max_reprojection_error);
    const bool good_depth_estimate =
        CheckDepthErrors(solution.depths[i], depths,
                         test_params.max_depth_sq_error);
    const double rotation_difference =
        expected_solution.rotation.angularDistance(solution.rotations[i]);
    const bool matched_rotation =
        rotation_difference < test_params.max_rotation_difference;
    const double translation_difference =
        (expected_solution.translation -
         solution.translations[i]).squaredNorm();
    const bool matched_translation =
        translation_difference < test_params.max_translation_difference;
    const double scale_difference =
        fabs(expected_solution.scale - solution.scales[i]);
    const bool matched_scale =
        scale_difference < test_params.max_scale_difference;
    VLOG(3) << "Matched rotation: " << matched_rotation
            << " rotation error [deg]=" << RadToDeg(rotation_difference);
    VLOG(3) << "Matched translation: " << matched_translation
            << " translation error=" << translation_difference;
    VLOG(3) << "Matched scale: " << matched_scale
            << " scale difference=" << scale_difference;
    VLOG(3) << "Good reprojection errors: " << good_reprojection_errors;
    VLOG(3) << "Estimated depths: " << solution.depths[i].transpose();
    VLOG(3) << "GT depths: " << depths.transpose();
    VLOG(3) << "Good depth estimate: " << good_depth_estimate;
    if (matched_rotation && matched_translation && matched_scale &&
        good_reprojection_errors && good_depth_estimate) {
      matched_transform = true;
    }
  }

  EXPECT_TRUE(matched_transform);
}

inline void Gp4pcTest::TestWithNoise(const ExpectedSolution& expected_solution,
                                     const TestParams& test_params,
                                     const TestInput& test_input) {
  TestWithNoise(expected_solution, test_params, test_input, false);
}

TEST_F(Gp4pcTest, BasicRigidTransformationTest) {
  TestInput test_input;
  // World points:
  test_input.world_points = {
    Eigen::Vector3d(-1.0, 3.0, 3.0),
    Eigen::Vector3d(1.0, -1.0, 2.0),
    Eigen::Vector3d(-1.0, 1.0, 2.0),
    Eigen::Vector3d(2.0, 1.0, 3.0)
  };

  // Camera centers.
  test_input.camera_centers = {
    Eigen::Vector3d(-1.0, 0.0, 0.0),
    Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d(2.0, 0.0, 0.0),
    Eigen::Vector3d(3.0, 0.0, 0.0)
  };

  // Test parameters.
  TestParams test_params;
  test_params.projection_noise_std_dev = 0.0;
  test_params.max_reprojection_error =  1.0 / 512;
  test_params.max_rotation_difference = DegToRad(1.0);
  test_params.max_translation_difference = 1e-3;
  test_params.max_scale_difference = 1e-3;
  test_params.max_depth_sq_error = 1e-3;

  // Expected solution.
  ExpectedSolution solution;
  solution.rotation = Quaterniond(AngleAxisd(
      DegToRad(21.0), Vector3d(0.5, 0.0, 1.0).normalized()));
  solution.translation = Vector3d(1.0, 1.0, 1.0);
  solution.scale = 1.0;

  TestWithNoise(solution, test_params, test_input);
}

TEST_F(Gp4pcTest, RigidTransformationTestWithNoise) {
  TestInput test_input;
  // World points:
  test_input.world_points = {
    Eigen::Vector3d(-1.0, 3.0, 3.0),
    Eigen::Vector3d(1.0, -1.0, 2.0),
    Eigen::Vector3d(-1.0, 1.0, 2.0),
    Eigen::Vector3d(2.0, 1.0, 3.0)
  };

  // Camera centers.
  test_input.camera_centers = {
    Eigen::Vector3d(-1.0, 0.0, 0.0),
    Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d(2.0, 0.0, 0.0),
    Eigen::Vector3d(3.0, 0.0, 0.0)
  };

  // Test parameters.
  TestParams test_params;
  test_params.projection_noise_std_dev = 1e-3;
  test_params.max_reprojection_error = 5.0 / 512;
  test_params.max_rotation_difference = DegToRad(1.0);
  test_params.max_translation_difference = 1e-2;
  test_params.max_scale_difference = 1e-2;
  test_params.max_depth_sq_error = 1e-2;

  // Expected solution.
  ExpectedSolution solution;
  solution.rotation = Quaterniond(AngleAxisd(
      DegToRad(21.0), Vector3d(0.5, 0.0, 1.0).normalized()));
  solution.translation = Vector3d(1.0, 1.0, 1.0);
  solution.scale = 1.0;

  TestWithNoise(solution, test_params, test_input);
}

TEST_F(Gp4pcTest, BasicSimilarityTransformationTest) {
  TestInput test_input;
  // World points:
  test_input.world_points = {
    Eigen::Vector3d(-1.0, 3.0, 3.0),
    Eigen::Vector3d(1.0, -1.0, 2.0),
    Eigen::Vector3d(-1.0, 1.0, 2.0),
    Eigen::Vector3d(2.0, 1.0, 3.0)
  };

  // Camera centers.
  test_input.camera_centers = {
    Eigen::Vector3d(-1.0, 0.0, 0.0),
    Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d(2.0, 0.0, 0.0),
    Eigen::Vector3d(3.0, 0.0, 0.0)
  };

  // Test parameters.
  TestParams test_params;
  test_params.projection_noise_std_dev = 0.0;
  test_params.max_reprojection_error =  1.0 / 512;
  test_params.max_rotation_difference = DegToRad(1.0);
  test_params.max_translation_difference = 1e-3;
  test_params.max_scale_difference = 1e-3;
  test_params.max_depth_sq_error = 1e-3;

  // Expected solution.
  ExpectedSolution solution;
  solution.rotation = Quaterniond(AngleAxisd(
      DegToRad(21.0), Vector3d(0.5, 0.0, 1.0).normalized()));
  solution.translation = Vector3d(1.0, 1.0, 1.0);
  solution.scale = 2.0;

  TestWithNoise(solution, test_params, test_input);
}

TEST_F(Gp4pcTest, SimilarityTransformationTestWithNoise) {
  TestInput test_input;
  // World points:
  test_input.world_points = {
    Eigen::Vector3d(-1.0, 3.0, 3.0),
    Eigen::Vector3d(1.0, -1.0, 2.0),
    Eigen::Vector3d(-1.0, 1.0, 2.0),
    Eigen::Vector3d(2.0, 1.0, 3.0)
  };

  // Camera centers.
  test_input.camera_centers = {
    Eigen::Vector3d(-1.0, 0.0, 0.0),
    Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d(2.0, 0.0, 0.0),
    Eigen::Vector3d(3.0, 0.0, 0.0)
  };

  // Test parameters.
  TestParams test_params;
  test_params.projection_noise_std_dev = 1e-3;
  test_params.max_reprojection_error =  5.0 / 512;
  test_params.max_rotation_difference = DegToRad(1.0);
  test_params.max_translation_difference = 1e-2;
  test_params.max_scale_difference = 1e-2;
  test_params.max_depth_sq_error = 1e-2;

  // Expected solution.
  ExpectedSolution solution;
  solution.rotation = Quaterniond(AngleAxisd(
      DegToRad(21.0), Vector3d(0.5, 0.0, 1.0).normalized()));
  solution.translation = Vector3d(1.0, 1.0, 1.0);
  solution.scale = 2.0;

  TestWithNoise(solution, test_params, test_input);
}

TEST_F(Gp4pcTest, BailOutOnSingleCameraCenter) {
  Input input;
  input.world_points = {
    Eigen::Vector3d(-5.2868, 13.6551, 37.7451),
    Eigen::Vector3d(0.907803, 9.2807, 42.1868),
    Eigen::Vector3d(3.80463, 14.9699, 42.3962),
    Eigen::Vector3d(0.0431288, 10.2583, 42.5212)
  };

  input.ray_directions = {
    Eigen::Vector3d(-0.334288, 00.265068, 00.904428),
    Eigen::Vector3d(0-0.37081, -0.479971, 00.795065),
    Eigen::Vector3d(-0.14836, 0.236824, 0.960158),
    Eigen::Vector3d(-0.26705, 0.138519, 0.953675)    
  };

  input.ray_origins = {
    Eigen::Vector3d(5.49048, -2.04053, -2.34362),
    Eigen::Vector3d(5.49048, -2.04053, -2.34362),
    Eigen::Vector3d(5.49048, -2.04053, -2.34362),
    Eigen::Vector3d(5.49048, -2.04053, -2.34362)
  };

  Eigen::Matrix3d gt_rotation;
  gt_rotation <<
      000.986208, 000.163803, 00.0237108,
      00-0.15569, 000.966737, 0-0.202928,
      -0.0561624, 000.196438, 000.978906;
  const Eigen::Vector3d translation(-1.3, 2.1, 00.5);
  const double scale = 2.5;

  // Estimate.
  Solution solution;
  TestParams test_params;
  Gp4pc solver(test_params.solver_params);
  EXPECT_FALSE(solver.EstimateSimilarityTransformation(input, &solution));
  EXPECT_TRUE(solution.rotations.empty());
  EXPECT_TRUE(solution.translations.empty());
  EXPECT_TRUE(solution.scales.empty());
}

TEST_F(Gp4pcTest, BasicPlanarCase) {
  TestInput test_input;
  // World points:
  test_input.world_points = {
    Eigen::Vector3d(-1.0, 3.0, 2.0),
    Eigen::Vector3d(1.0, -1.0, 2.0),
    Eigen::Vector3d(-1.0, 1.0, 2.0),
    Eigen::Vector3d(2.0, 1.0, 2.0)
  };

  // Camera centers.
  test_input.camera_centers = {
    Eigen::Vector3d(-1.0, 0.0, 0.0),
    Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d(2.0, 0.0, 0.0),
    Eigen::Vector3d(3.0, 0.0, 0.0)
  };

  // Test parameters.
  TestParams test_params;
  test_params.projection_noise_std_dev = 0.0;
  test_params.max_reprojection_error =  1.0 / 512;
  test_params.max_rotation_difference = DegToRad(1.0);
  test_params.max_translation_difference = 1e-3;
  test_params.max_scale_difference = 1e-3;
  test_params.max_depth_sq_error = 1e-3;
  // Set false to use_general_solver to allow the planar solver to be used.
  test_params.solver_params.use_general_solver = false;

  // Expected solution.
  ExpectedSolution solution;
  solution.rotation = Quaterniond(AngleAxisd(
      DegToRad(21.0), Vector3d(0.5, 0.0, 1.0).normalized()));
  solution.translation = Vector3d(1.0, 1.0, 1.0);
  solution.scale = 2.0;

  TestWithNoise(solution, test_params, test_input, true);
}


}  // namespace
}  // namespace msft
