# Require a CMake version providing FindCUDAToolkit
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

## Only policies introduced after the cmake_minimum_required
## version need to explicitly be set to NEW.
## Refer to https://cmake.org/cmake/help/v3.11/manual/cmake-policies.7.html
set(CMAKE_POLICIES CMP0135)

foreach(p ${CMAKE_POLICIES})
  if(POLICY ${p})
    cmake_policy(SET ${p} NEW)
  endif()
endforeach()

project(CudaCommon)

## CudaCommon Version
set(CudaCommon_VERSION_MAJOR "2")
set(CudaCommon_VERSION_MINOR "0")
set(CudaCommon_VERSION_PATCH "0")
set(
  CudaCommon_VERSION_STRING
  "${CudaCommon_VERSION_MAJOR}.${CudaCommon_VERSION_MINOR}"
)

#=========================================================
# Installation variables
#=========================================================
if(NOT CudaCommon_INSTALL_INCLUDE_DIR)
  set(CudaCommon_INSTALL_INCLUDE_DIR include/CudaCommon)
endif()

if(NOT ITK_SOURCE_DIR)
  include(itk-module-init.cmake)
endif()

set(CudaCommon_LIBRARIES CudaCommon)
set(
  CudaCommon_INCLUDE_DIRS
  ${CudaCommon_SOURCE_DIR}/include
  ${CUDAToolkit_INCLUDE_DIRS}
)

# --------------------------------------------------------
# Find ITK (required)
if(NOT ITK_SOURCE_DIR)
  find_package(ITK 5.2 REQUIRED)
endif()

# Propagate cmake options in a header file
configure_file(
  ${CudaCommon_SOURCE_DIR}/itkCudaCommonConfiguration.h.in
  ${CudaCommon_BINARY_DIR}/include/itkCudaCommonConfiguration.h
)

list(APPEND CudaCommon_INCLUDE_DIRS ${CudaCommon_BINARY_DIR})

set(
  CudaCommon_EXPORT_CODE_BUILD
  "
# Pass source dir to allow other modules, e.g. RTK, to use CudaImage.i.init and CudaImage.i.in
set(CudaCommon_SOURCE_DIR ${CudaCommon_SOURCE_DIR})

# Also pass CudaCommon version
set(CudaCommon_VERSION_MAJOR ${CudaCommon_VERSION_MAJOR})
set(CudaCommon_VERSION_MINOR ${CudaCommon_VERSION_MINOR})
set(CudaCommon_VERSION_PATCH ${CudaCommon_VERSION_PATCH})
set(CudaCommon_VERSION_STRING ${CudaCommon_VERSION_STRING})
"
)

if(NOT ITK_SOURCE_DIR)
  if(NOT EXISTS ${ITK_CMAKE_DIR}/ITKModuleMacros.cmake)
    message(
      FATAL_ERROR
      "Modules can only be built against an ITK build tree; they cannot be built against an ITK install tree."
    )
  endif()

  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleExternal)
else()
  set(ITK_DIR ${CMAKE_BINARY_DIR})
  itk_module_impl()
endif()

# Install configuration file
install(
  FILES
    ${CudaCommon_BINARY_DIR}/include/itkCudaCommonConfiguration.h
  DESTINATION ${CudaCommon_INSTALL_INCLUDE_DIR}
)
