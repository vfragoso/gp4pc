// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// gDLS*: Generalized Pose-and-Scale Estimation Given Scale and Gravity Priors
//
// Victor Fragoso, Joseph DeGol, Gang Hua.
// Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition 2020.
//
// Please contact the author of this library if you have any questions.
// Author: Victor Fragoso (victor.fragoso@microsoft.com)

#include <stdlib.h>
#include <cmath>
#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gp4pc/camera_feature_correspondence_2d_3d.h>
#include <gp4pc/estimate_similarity_transformation.h>
#include <gp4pc/pinhole_camera.h>
#include <math/utils.h>

namespace msft {
namespace {

// Random seed.
constexpr size_t kSeed = 67;
// Focal length.
static const double kFocalLength = 1000.0;
// Number of cameras.
static const int kNumCameras = 4;
// Number of points.
static const int kNumPoints = 100;
// Min. angle in degrees.
static const double kMinAngleDeg = 5.0;
// Max. angle in degrees.
static const double kMaxAngleDeg = 40.0;
// Max. num of trials.
static const int kNumTrials = 8;
// Minimum and maximum limits for generating 3D points.
static const double kMinXY = -5.0;
static const double kMaxXY = 5.0;
static const double kMinZ = 10;
static const double kMaxZ = 20;

// Similarity transformation.
struct SimilarityTransformation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  double scale = 1.0;
};

struct TestParameters {
  // Num. of points for the test.
  int num_points = 4;
  // Noise level to points in 3d.
  double noise_std_dev = 0.0;
  // Inlier ratio. 1.0: All inliers, 0.0: All outliers.
  double inlier_ratio = 1.0;
  // Rotation error threshold in radians.
  double rotation_thresh = DegToRad(1.0);
  // Translation error threshold.
  double translation_thresh = 0.5;
  // Scale error threshold.
  double scale_thresh = 0.1;
  // Rotation generation parameters.
  double min_angle_deg = 1.0;
  double max_angle_deg = 45.0;
  // Translation generation parameters.
  double min_translation_value = 0.0;
  double max_translation_value = 10.0;
  // Scale generation parameters.
  double min_scale_value = 0.5;
  double max_scale_value = 5.0;
  // Ransac parameters.
  Gp4pcRobustEstimator::RansacParameters ransac_params;
};

class EstimateSimilarityTransformationTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    rng = new std::mt19937(kSeed);
    srand(kSeed);
  }

  static void TearDownTestCase() {
    delete rng;
  }

  double GenerateRandomScale(const double min_value,
                             const double max_value) {
    CHECK_LT(min_value, max_value);
    std::mt19937& prng = *rng;
    std::uniform_real_distribution<double> dist(min_value, max_value);
    return dist(prng);
  }

  Eigen::Vector3d GenerateRandomVector3d(const double min_value,
                                         const double max_value) {
    CHECK_LT(min_value, max_value);
    std::mt19937& prng = *rng;
    std::uniform_real_distribution<double> dist(min_value, max_value);
    const Eigen::Vector3d point(dist(prng),
                                dist(prng),
                                dist(prng));
    return point;
  }

  Eigen::Quaterniond GenerateRandomRotation(const double min_angle_deg,
                                            const double max_angle_deg) {
    CHECK_LT(min_angle_deg, max_angle_deg);
    std::uniform_real_distribution<double> angle_dist(min_angle_deg,
                                                      max_angle_deg);
    std::mt19937& prng = *rng;
    const Eigen::Quaterniond rotation(
        Eigen::AngleAxisd(DegToRad(angle_dist(prng)),
                          Eigen::Vector3d::Random().normalized()));
    return rotation;
  }

  PinholeCamera GenerateRandomCamera() {
    const Eigen::Vector2d principal_point(kFocalLength / 2.0,
                                          kFocalLength / 2.0);
    const Eigen::Quaterniond rotation = GenerateRandomRotation(kMinAngleDeg,
                                                               kMaxAngleDeg);
    const Eigen::Vector3d translation = GenerateRandomVector3d(-10, 10);
    PinholeCamera camera(kFocalLength, principal_point, rotation, translation);
    return camera;
  }

  Eigen::Vector3d GenerateRandomPoint3d(const double min_xy,
                                        const double max_xy,
                                        const double min_z,
                                        const double max_z) {
    std::mt19937& prng = *rng;
    std::uniform_real_distribution<double> xy_dist(min_xy, max_xy);
    std::uniform_real_distribution<double> z_dist(min_z, max_z);
    Eigen::Vector3d point(xy_dist(prng), xy_dist(prng), z_dist(prng));
    return point;
  }

  Eigen::Vector3d AddNoiseToPoint3d(const double noise_std_dev,
                                    Eigen::Vector3d* point) {
    std::mt19937& prng = *rng;
    std::normal_distribution<double> noise(0.0, noise_std_dev);
    *CHECK_NOTNULL(point) += Eigen::Vector3d(noise(prng),
                                             noise(prng),
                                             noise(prng));
    return *point;
  }

  SimilarityTransformation GenerateRandomSimilarityTransformation(
      const TestParameters& test_params) {
    SimilarityTransformation transformation;
    transformation.rotation = GenerateRandomRotation(test_params.min_angle_deg,
                                                     test_params.max_angle_deg);
    transformation.translation =
        GenerateRandomVector3d(test_params.min_translation_value,
                               test_params.max_translation_value);
    transformation.scale = GenerateRandomScale(test_params.min_scale_value,
                                               test_params.max_scale_value);
    return transformation;
  }

  std::vector<CameraFeatureCorrespondence2D3D> GenerateCorrespondences(
      const TestParameters& test_params,
      const SimilarityTransformation& expected_transformation) {
    CHECK_GT(test_params.inlier_ratio, 0.0);
    CHECK_LE(test_params.inlier_ratio, 1.0);
    std::vector<CameraFeatureCorrespondence2D3D> correspondences(
        test_params.num_points);

    // Generate cameras.
    std::vector<PinholeCamera> cameras(kNumCameras);
    for (int i = 0; i < kNumCameras; ++i) {
      cameras[i] = GenerateRandomCamera();
    }

    // Generate synthetic correspondences.
    for (int i = 0; i < test_params.num_points; ++i) {
      CameraFeatureCorrespondence2D3D& correspondence = correspondences[i];
      correspondence.camera = cameras[i % cameras.size()];
      double depth = -1;
      do {
        // Create a random 3D point that is in front of the camera.
        correspondence.point =
            GenerateRandomPoint3d(kMinXY, kMaxXY, kMinZ, kMaxZ);
        // Project point and see if it is infront of camera.
        depth = correspondence.camera.ProjectPoint(correspondence.point,
                                                   &correspondence.observation);
      } while(depth < 0);

      // Add zero-mean Gaussian noise.
      if (test_params.noise_std_dev > 0.0) {
        AddNoiseToPoint3d(test_params.noise_std_dev, &correspondence.point);
      }
    }

    // Add outliers and apply inv. transformation.
    for (int i = 0; i < correspondences.size(); ++i) {
      // Add outlier.
      if (i > test_params.inlier_ratio * correspondences.size()) {
        correspondences[i].observation =
            kFocalLength * Eigen::Vector2d::Random();
      }

      // Apply inv. transformation.
      const Eigen::Vector3d new_point =
          expected_transformation.rotation.conjugate() *
          (expected_transformation.scale * correspondences[i].point -
           expected_transformation.translation);
      correspondences[i].point = std::move(new_point);
    }

    return correspondences;
  }

  bool EvaluateSolution(const TestParameters& test_params,
                        const SimilarityTransformation& expected_transformation,
                        const Gp4pc::Solution& estimated_transformations) {
    // Check that we get a single solution.
    EXPECT_GE(estimated_transformations.rotations.size(), 1);
    EXPECT_GE(estimated_transformations.rotations.size(),
              estimated_transformations.translations.size());
    EXPECT_GE(estimated_transformations.rotations.size(),
              estimated_transformations.scales.size());

    // Estimated solution.
    const int num_solutions = estimated_transformations.rotations.size();
    const Eigen::Quaterniond& expected_rotation =
        expected_transformation.rotation;
    const Eigen::Vector3d& expected_translation =
        expected_transformation.translation;
    const double& expected_scale = expected_transformation.scale;
    int num_trials_passed = 0;
    // Evaluation of solutions.
    for (int i = 0; i < num_solutions; ++i) {
      const Eigen::Quaterniond& estimated_rotation =
          estimated_transformations.rotations[i];
      const Eigen::Vector3d& estimated_translation =
          estimated_transformations.translations[i];
      const double& estimated_scale = estimated_transformations.scales[i];
      // Compute errors.
      const double rotation_error =
          expected_rotation.angularDistance(estimated_rotation);
      const double translation_error =
          (expected_translation - estimated_translation).norm();
      const double scale_error = std::abs(expected_scale - estimated_scale);

      const bool good_rotation = rotation_error < test_params.rotation_thresh;
      const bool good_translation =
           translation_error < test_params.translation_thresh;
      const bool good_scale = scale_error < test_params.scale_thresh;
      const bool good_run = good_rotation && good_translation && good_scale;

      VLOG_IF(3, !good_run) << "Solution: " << (i + 1) << "/" << num_solutions
                            << "\nRotation error: " << rotation_error
                            << " thresh: " << test_params.rotation_thresh
                            << " good: " << good_rotation
                            << "\nTranslation error: " << translation_error
                            << " thresh: " << test_params.translation_thresh
                            << " good: " << good_translation
                            << "\nScale error: " << scale_error
                            << " thresh: " << test_params.scale_thresh
                            << " good: " << good_scale;

      if (good_run) {
        ++num_trials_passed;
      }
    }

    return num_trials_passed > 0;
  }

  void ExecuteRandomTest(
      const TestParameters& test_params,
      const SimilarityTransformation& expected_transformation) {
    // Compute 2D-3D correspondences.
    const std::vector<CameraFeatureCorrespondence2D3D> correspondences =
        GenerateCorrespondences(test_params, expected_transformation);

    // Estimate similarity transformation.
    const Gp4pc::Solution solution =
        EstimateSimilarityTransformation(correspondences);

    // Evaluate solution.
    EXPECT_TRUE(EvaluateSolution(test_params,
                                 expected_transformation,
                                 solution));
  }


  void ExecuteRansacRandomTest(
      const TestParameters& test_params,
      const SimilarityTransformation& expected_transformation) {
    // Compute 2D-3D correspondences.
    const std::vector<CameraFeatureCorrespondence2D3D> correspondences =
        GenerateCorrespondences(test_params, expected_transformation);

    // Estimate similarity transformation.
    Gp4pcRobustEstimator::RansacSummary ransac_summary;
    const Gp4pc::Solution solution =
        EstimateSimilarityTransformation(test_params.ransac_params,
                                         correspondences,
                                         &ransac_summary);

    // Evaluate solution.
    EXPECT_TRUE(EvaluateSolution(test_params,
                                 expected_transformation,
                                 solution));
  }

  // Random number generator.
  static std::mt19937* rng;
};

std::mt19937* EstimateSimilarityTransformationTest::rng = nullptr;

// Minimal estimation tests.
TEST_F(EstimateSimilarityTransformationTest,
       MinimalBasicEstimationNoNoise) {
  TestParameters test_params;
  for (int i = 0; i < kNumTrials; ++i) {
    const SimilarityTransformation expected_transformation =
        GenerateRandomSimilarityTransformation(test_params);
    ExecuteRandomTest(test_params, expected_transformation);
  }
}

TEST_F(EstimateSimilarityTransformationTest,
       MinimalBasicEstimationWithNoise) {
  TestParameters test_params;
  test_params.noise_std_dev = 1e-3;
  test_params.rotation_thresh = DegToRad(5.0);
  test_params.translation_thresh = 1.0;
  test_params.scale_thresh = 0.5;
  for (int i = 0; i < kNumTrials; ++i) {
    const SimilarityTransformation expected_transformation =
        GenerateRandomSimilarityTransformation(test_params);
    ExecuteRandomTest(test_params, expected_transformation);
  }
}

// Non-minimal estimation tests.
TEST_F(EstimateSimilarityTransformationTest,
       NonMinimalBasicEstimationNoNoise) {
  TestParameters test_params;
  test_params.num_points = kNumPoints;
  for (int i = 0; i < kNumTrials; ++i) {
    const SimilarityTransformation expected_transformation =
        GenerateRandomSimilarityTransformation(test_params);
    ExecuteRandomTest(test_params, expected_transformation);
  }
}

TEST_F(EstimateSimilarityTransformationTest,
       NonMinimalBasicEstimationWithNoise) {
  TestParameters test_params;
  test_params.num_points = kNumPoints;
  test_params.noise_std_dev = 1e-3;
  test_params.rotation_thresh = DegToRad(5.0);
  test_params.translation_thresh = 1.0;
  test_params.scale_thresh = 0.5;
  for (int i = 0; i < kNumTrials; ++i) {
    const SimilarityTransformation expected_transformation =
        GenerateRandomSimilarityTransformation(test_params);
    ExecuteRandomTest(test_params, expected_transformation);
  }
}

// Test RANSAC estimators all inliers.
TEST_F(EstimateSimilarityTransformationTest, AllInliersWithNoise) {
  TestParameters test_params;
  test_params.num_points = kNumPoints;
  test_params.noise_std_dev = 1e-3;
  test_params.rotation_thresh = DegToRad(5.0);
  test_params.translation_thresh = 1.0;
  test_params.scale_thresh = 0.5;
  for (int i = 0; i < kNumTrials; ++i) {
    const SimilarityTransformation expected_transformation =
        GenerateRandomSimilarityTransformation(test_params);
    ExecuteRansacRandomTest(test_params, expected_transformation);
  }
}

// RANSAC w/ inliers and outliers and noise.
TEST_F(EstimateSimilarityTransformationTest,
       InliersAndOutliersWithNoise) {
  TestParameters test_params;
  test_params.num_points = kNumPoints;
  test_params.noise_std_dev = 1e-3;
  test_params.rotation_thresh = DegToRad(5.0);
  test_params.translation_thresh = 1.0;
  test_params.scale_thresh = 0.5;
  test_params.inlier_ratio = 0.8;
  for (int i = 0; i < kNumTrials; ++i) {
    const SimilarityTransformation expected_transformation =
        GenerateRandomSimilarityTransformation(test_params);
    ExecuteRansacRandomTest(test_params, expected_transformation);
  }
}

}  // namespace
}  // namespace msft

