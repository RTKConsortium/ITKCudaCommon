#
# Find the packages required by this module
#

# Set default Cuda architecture if not provided. The first case allows for
# backward compatibility with cmake versions before 3.20 which did not handle
# CUDAARCHS environment variable.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(DEFINED ENV{CUDAARCHS})
    set(
      CMAKE_CUDA_ARCHITECTURES
      "$ENV{CUDAARCHS}"
      CACHE STRING
      "CUDA architectures"
    )
  else()
    set(CMAKE_CUDA_ARCHITECTURES "52" CACHE STRING "CUDA architectures")
  endif()
endif()
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  set(CUDAToolkit_NVCC_EXECUTABLE ${CMAKE_CUDA_COMPILER})
  find_package(CUDAToolkit REQUIRED)
  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDAToolkit not found (available since CMake v3.17)")
  endif()
endif()

# Configure CUDA compilation options
if( NOT CUDACOMMON_CUDA_VERSION)
  set(CUDACOMMON_CUDA_VERSION ${CUDAToolkit_VERSION} CACHE STRING "Specify the exact CUDA version that must be used for CudaCommon")
else()
  set(CUDACOMMON_CUDA_VERSION ${CUDACOMMON_CUDA_VERSION} CACHE STRING "Specify the exact CUDA version that must be used for CudaCommon")
endif()
mark_as_advanced(CUDACOMMON_CUDA_VERSION)

if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
  if(CUDACOMMON_CUDA_VERSION)
    find_package(CUDAToolkit EXACT ${CUDACOMMON_CUDA_VERSION} REQUIRED)
  endif()
endif()
