# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Generalized Pose-and-Scale Estimation using 4-Point Congruence Constraints
#
# Victor Fragoso and Sudipta Sinha.
# In Proc. of the IEEE International Conf. on 3D Vision (3DV) 2020.
#
# Please contact the author of this library if you have any questions.
# Author: Victor Fragoso (victor.fragoso@microsoft.com)

cmake_minimum_required(VERSION 3.1)
project(gp4pc C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif (COMMAND cmake_policy)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
option(BUILD_TESTING "Enable testing" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Optimize compiler flags.
include(OptimizeCompilerFlags)
optimizecompilerflags()

# Include checklibraryexists.
include(CheckLibraryExists)
CHECK_LIBRARY_EXISTS(m sin "" HAVE_LIB_M)
if (HAVE_LIB_M)
    set(EXTRA_LIBS ${LIBM} m)
endif (HAVE_LIB_M)

# Find Eigen.
message("-- Check for Eigen.")
set(MIN_EIGEN_VERSION 3.2.0)
find_package(Eigen ${MIN_EIGEN_VERSION} REQUIRED)
if (EIGEN_FOUND)
  message("-- Found Eigen version ${EIGEN_VERSION}: ${EIGEN_INCLUDE_DIRS}")
endif (EIGEN_FOUND)

# Find Glog.
message("-- Check for Glog.")
set(GLOG_PREFER_EXPORTED_GLOG_CMAKE_CONFIGURATION FALSE)
find_package(Glog REQUIRED)
if (GLOG_FOUND)
  message("-- Found Glog [include]: ${GLOG_INCLUDE_DIRS}")
  message("-- Found Glog [library]: ${GLOG_LIBRARY}")
  message("-- CMake Glog Config: ${FOUND_INSTALLED_GLOG_CMAKE_CONFIGURATION}")
else (GLOG_FOUND)
  message(FATAL_ERROR "Can't find Glog. Please set GLOG_INCLUDE_DIR & "
    "GLOG_LIBRARY")
endif (GLOG_FOUND)

# Find Gflags.
message("-- Check for Gflags.")
find_package(gflags REQUIRED)
if (gflags_FOUND)
  message("-- Found Google Flags [include]: ${gflags_INCLUDE_DIR}")
endif (gflags_FOUND)

# Defining variable for all the including directories.
set(GP4PC_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/src
  ${EIGEN_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_DIR})

# Defining variable for gDLS* library dependencies.
set(GP4PC_LIBRARY_DEPS ${LIBM} ${GLOG_LIBRARY} gflags)

# Find gtest for building unit tests.
if (BUILD_TESTING)
  enable_testing()
  # Download and unpack googletest at configure time.
  # Snippet of code from:
  # https://github.com/google/googletest/blob/master/googletest/README.md
  configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/GoogleTest.CmakeLists.txt.in
    googletest-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
  if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
  if (result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
  endif()

  # Prevent overriding the parent project's compiler/linker
  # settings on Windows
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  # Add googletest directly to our build. This defines
  # the gtest and gtest_main targets.
  add_subdirectory(
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-src
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
    EXCLUDE_FROM_ALL)

endif()

# Include dirs.
include_directories(${GP4PC_INCLUDE_DIRS})

# Src. files.
set(GP4PC_SRC
  src/gp4pc/estimate_similarity_transformation.cc
  src/gp4pc/gp4pc.cc
  src/gp4pc/gp4pc_robust_estimator.cc
  src/gp4pc/pinhole_camera.cc
  src/gp4pc/util.cc
  src/math/solve_gp4pc_polynomial.cc
  src/third_party/align_point_clouds.cc)

# Compile gdls* library.
add_library(gp4pc STATIC ${GP4PC_SRC})
target_link_libraries(gp4pc ${GP4PC_LIBRARY_DEPS})

# Compile unit tests.
if (BUILD_TESTING)
  macro (TEST FILENAME)
    string(REPLACE "/" ";" TEST_NAME_LIST ${FILENAME})
    list(REVERSE TEST_NAME_LIST)
    list(GET TEST_NAME_LIST 0 TEST_NAME)
    add_executable(${TEST_NAME}_test test/test_main.cc ${FILENAME}_test.cc)
    target_link_libraries(${TEST_NAME}_test
      gtest
      ${GDLS_STAR_LIBRARY_DEPS}
      gp4pc)
    add_test(NAME ${TEST_NAME}_test
      COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_NAME}_test)
  endmacro (TEST)

  test(src/gp4pc/estimate_similarity_transformation)
  # test(src/gp4pc/gp4pc)
  test(src/gp4pc/pinhole_camera)
  test(src/math/solve_gp4pc_polynomial)

endif(BUILD_TESTING)