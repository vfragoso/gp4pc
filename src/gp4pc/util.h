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

#ifndef GP4PC_UTIL_H_
#define GP4PC_UTIL_H_

#include <vector>
#include <gp4pc/camera_feature_correspondence_2d_3d.h>
#include <gp4pc/gp4pc.h>

namespace msft {

// Computes the gDLS* input from the 2D-3D correspondences and its respective
// camera observing the 3D point.
Gp4pc::Input ComputeInputDatum(
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences);

}  // namespace msft

#endif  // GP4PC_UTIL_H_
