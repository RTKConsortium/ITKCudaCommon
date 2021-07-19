#
# Find the packages required by this module
#
include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
  find_package(CUDAToolkit REQUIRED)
  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDAToolkit not found (available since CMake v3.17)")
  endif()
endif()

# Configure CUDA compilation options
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
endif()
