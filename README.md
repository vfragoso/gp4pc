# gp4pc: Generalized Pose-and-Scale Estimation using 4-Point Congruence Constraints

### Authors:
### Victor Fragoso (victor.fragoso@microsoft.com)
### Sudipta Sinha (sudipta.sinha@microsoft.com)

## Citation

If you use this code for your research, please cite the following paper:
```
@inproceedings{Fragoso_Sinha_2020_3DV,
author = {Fragoso, Victor and Sinha, Sudipta},
title = {Generalized Pose-and-Scale Estimation using 4-Point Congruence Constraints},
booktitle = {Proc. of the IEEE International Conf. on 3D Vision (3DV)},
month = {November},
year = {2020}
}
```

[PDF](https://arxiv.org/pdf/2011.13817.pdf)

## Video
YouTube: https://youtu.be/RJTv8rKr-PI

## Compilation

### Pre-requisites

- CMake [v3.1 or greater] (https://cmake.org/)
- Eigen3 (http://eigen.tuxfamily.org/)
- Glog (https://github.com/google/glog)
- Gflags (https://gflags.github.io/gflags/)

### Steps
 1. Install CMake.
 2. Install Eigen3 - this is a C++ header library.
 3. Install Gflags.
 4. Install Glog.
 5. Run cmake to generate building scripts or solutions.
 6. Invoke the cmake-generated building scripts (e.g., Makefiles or VS solutions).

### Optional components
 1. UnitTest: The project can build unit-tests. To do so, enable the CMake flag BUILD_TESTING.

*NOTE*: When building the optional components, the cmake script will take care of downloading and compiling the dependencies.

### License
Read more [here](./LICENSE.txt)

### Contributing
Read more [here](./CONTRIBUTING.md)

### Microsoft Code of Conduct
Read more [here](https://opensource.microsoft.com/codeofconduct)

### Trademarks Note
Trademarks This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.