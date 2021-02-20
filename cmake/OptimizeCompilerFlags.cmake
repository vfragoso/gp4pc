# Most of the compilation flags are coming from Google-Ceres:
#   https://github.com/ceres-solver/ceres-solver/blob/master/CMakeLists.txt
#
# and Theia-SfM library:
#   https://github.com/sweeneychris/TheiaSfM/blob/master/cmake/OptimizeTheiaCompilerFlags.cmake
#
# Modified by Victor Fragoso (victor.fragoso@microsoft.com)
#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# gDLS*: Generalized Pose-and-Scale Estimation Given Scale and Gravity Priors
#
# Victor Fragoso, Joseph DeGol, Gang Hua.
# Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition 2020.
#
# Please contact the author of this library if you have any questions.
# Author: Victor Fragoso (victor.fragoso@microsoft.com)
#   
# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2015 Google Inc. All rights reserved.
# http://ceres-solver.org/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Authors: keir@google.com (Keir Mierle)
#          alexs.mac@gmail.com (Alex Stewart)
#
# Copyright (C) 2013 The Regents of the University of California (Regents).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of The Regents or University of California nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Please contact the author of this library if you have any questions.
# Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

macro(OptimizeCompilerFlags)
  # Set build type to Release if possible to ensure performance.
  if (NOT CMAKE_BUILD_TYPE)
    message("-- No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose a build type: None, Debug, Release, RelWithDebInfo, MinSizeRel." FORCE)
  else (NOT CMAKE_BUILD_TYPE)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      message("\n<<<< WARNING >>>>")
      message("\n-- Build type: Debug. Performance will be affected!")
      message("\n-- Set -DCMAKE_BUILD_TYPE=Release to get a performant build.")
      message("\n<<<< WARNING >>>>")
    endif (CMAKE_BUILD_TYPE STREQUAL "Debug")
  endif (NOT CMAKE_BUILD_TYPE)

  if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      # Use of -O3 requires use of gold linker & LLVM-gold plugin, which might
      # well not be present / in use and without which files will compile, but
      # not link ('file not recognized') so explicitly check for support
      include(CheckCXXCompilerFlag)
      check_cxx_compiler_flag("-flto" HAVE_LTO_SUPPORT)
      if (HAVE_LTO_SUPPORT)
        message(STATUS "Enabling link-time optimization (-flto)")
        set(THEIA_CXX_FLAGS "${THEIA_CXX_FLAGS} -flto")
      else ()
        message(STATUS "Compiler/linker does not support link-time optimization (-flto), disabling.")
      endif (HAVE_LTO_SUPPORT)
    endif ()
  endif()
  
  if(MSVC)
    # On MSVC, math constants are not included in <cmath> or <math.h> unless
    # _USE_MATH_DEFINES is defined [1].  As we use M_PI in the examples, ensure
    # that _USE_MATH_DEFINES is defined before the first inclusion of <cmath>.
    #
    # [1] https://msdn.microsoft.com/en-us/library/4hwaceh6.aspx
    add_definitions("-D_USE_MATH_DEFINES")
    # Disable signed/unsigned int conversion warnings.
    add_compile_options("/wd4018" "/wd4267")
    # Disable warning about using struct/class for the same symobl.
    add_compile_options("/wd4099")
    # Disable warning about the insecurity of using "std::copy".
    add_compile_options("/wd4996")
    # Disable performance warning about int-to-bool conversion.
    add_compile_options("/wd4800")
    # Disable performance warning about fopen insecurity.
    add_compile_options("/wd4996")
    # Disable warning about int64 to int32 conversion. Disabling
    # this warning may not be correct; needs investigation.
    # TODO(keir): Investigate these warnings in more detail.
    add_compile_options("/wd4244")
    # It's not possible to use STL types in DLL interfaces in a portable and
    # reliable way. However, that's what happens with Google Log and Google Flags
    # on Windows. MSVC gets upset about this and throws warnings that we can't do
    # much about. The real solution is to link static versions of Google Log and
    # Google Test, but that seems tricky on Windows. So, disable the warning.
    add_compile_options("/wd4251")

    # Add bigobj flag otherwise the build would fail due to large object files
    # probably resulting from generated headers (like the fixed-size schur
    # specializations).
    add_compile_options("/bigobj")

    # Google Flags doesn't have their DLL import/export stuff set up correctly,
    # which results in linker warnings. This is irrelevant for Ceres, so ignore
    # the warnings.
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4049")

    # Update the C/CXX flags for MSVC to use either the static or shared
    # C-Run Time (CRT) library based on the user option: MSVC_USE_STATIC_CRT.
    list(APPEND C_CXX_FLAGS
      CMAKE_CXX_FLAGS
      CMAKE_CXX_FLAGS_DEBUG
      CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL
      CMAKE_CXX_FLAGS_RELWITHDEBINFO)

    foreach(FLAG_VAR ${C_CXX_FLAGS})
      if (MSVC_USE_STATIC_CRT)
        # Use static CRT.
        if (${FLAG_VAR} MATCHES "/MD")
          string(REGEX REPLACE "/MD" "/MT" ${FLAG_VAR} "${${FLAG_VAR}}")
        endif (${FLAG_VAR} MATCHES "/MD")
      else (MSVC_USE_STATIC_CRT)
        # Use shared, not static, CRT.
        if (${FLAG_VAR} MATCHES "/MT")
          string(REGEX REPLACE "/MT" "/MD" ${FLAG_VAR} "${${FLAG_VAR}}")
        endif (${FLAG_VAR} MATCHES "/MT")
      endif (MSVC_USE_STATIC_CRT)
    endforeach()
  
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")

    # Tuple sizes of 10 are used by Gtest.
    add_definitions("-D_VARIADIC_MAX=10")
  endif()

  if (UNIX)
    set(COMMON_CXX_FLAGS "-Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native")
  endif()
    
  # Use a larger inlining threshold for Clang, since it hobbles Eigen,
  # resulting in an unreasonably slow version of the blas routines. The
  # -Qunused-arguments is needed because CMake passes the inline
  # threshold to the linker and clang complains about it and dies.
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang") # Matches Clang & AppleClang.
    set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Qunused-arguments -mllvm -inline-threshold=600")
      
    # Older versions of Clang (<= 2.9) do not support the 'return-type-c-linkage'
    # option, so check for its presence before adding it to the default flags set.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-Wno-return-type-c-linkage"
      HAVE_RETURN_TYPE_C_LINKAGE)
    if (HAVE_RETURN_TYPE_C_LINKAGE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-return-type-c-linkage")
    endif(HAVE_RETURN_TYPE_C_LINKAGE)
  endif ()

endmacro(OptimizeCompilerFlags)
