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

#ifndef GP4PC_SOLVE_GP4PC_POLYNOMIAL_H_
#define GP4PC_SOLVE_GP4PC_POLYNOMIAL_H_

#include <Eigen/Dense>

namespace msft {

// Solves for the gP4P+s problem using congruency of tetrahedrons.
// Params:
//   coeff_mat:  Coefficient matrix. Must be 4 x 15 matrix.
Eigen::MatrixXcd SolveGp4pcPolynomial(const Eigen::MatrixXd& coeff_mat);

}  // namespace msft


#endif  // GP4PC_SOLVE_GP4PC_POLYNOMIAL_H_
