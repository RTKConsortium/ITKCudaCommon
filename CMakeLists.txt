# Require a CMake version providing FindCUDAToolkit
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(CudaCommon)

if(NOT ITK_SOURCE_DIR)
  include(itk-module-init.cmake)
endif()

set(CudaCommon_LIBRARIES CudaCommon)
set(CudaCommon_INCLUDE_DIRS ${CudaCommon_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})

# --------------------------------------------------------
# Find ITK (required)
if(NOT ITK_SOURCE_DIR)
  find_package(ITK 5.2 REQUIRED)
endif()

if(NOT ITK_SOURCE_DIR)
  if(NOT EXISTS ${ITK_CMAKE_DIR}/ITKModuleMacros.cmake)
    message(FATAL_ERROR "Modules can only be built against an ITK build tree; they cannot be built against an ITK install tree.")
  endif()

  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleExternal)
else()
  set(ITK_DIR ${CMAKE_BINARY_DIR})
  itk_module_impl()
endif()
