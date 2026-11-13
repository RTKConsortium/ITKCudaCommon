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
set(CudaCommon_VERSION_MINOR "1")
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

if(NOT ITK_SOURCE_DIR)
  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
endif()
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
if(GIT_SHA1 MATCHES ".*NOTFOUND")
  set(CudaCommon_VERSION_HASH "")
else()
  set(CudaCommon_VERSION_HASH ", git hash ${GIT_SHA1}")
endif()

# Pixel types used for CUDA operator instantiations and Python wrapping.
set(
  CudaCommon_OPERATOR_INSTANTIATIONS_TYPES
  "float;double;int;unsigned int;unsigned long;short;unsigned short;unsigned char"
  CACHE STRING
  "C++ pixel types instantiated and wrapped for CUDA operators (float is always instantiated)"
)
# float is the library baseline and is always available.
list(APPEND CudaCommon_OPERATOR_INSTANTIATIONS_TYPES float)
list(REMOVE_DUPLICATES CudaCommon_OPERATOR_INSTANTIATIONS_TYPES)

set(CudaCommon_OPERATOR_INSTANTIATION_DEFINITIONS "")
foreach(type IN LISTS CudaCommon_OPERATOR_INSTANTIATIONS_TYPES)
  if(CudaCommon_OPERATOR_INSTANTIATION_DEFINITIONS)
    string(APPEND CudaCommon_OPERATOR_INSTANTIATION_DEFINITIONS " \\\n  ")
  endif()
  string(
    APPEND
    CudaCommon_OPERATOR_INSTANTIATION_DEFINITIONS
    "template decltype(OP<${type}>) CudaCommon_EXPORT OP<${type}>;"
  )
endforeach()

# Real-only operators use the floating-point subset of the same list.
set(
  CudaCommon_REAL_OPERATOR_INSTANTIATIONS_TYPES
  "${CudaCommon_OPERATOR_INSTANTIATIONS_TYPES}"
)
list(
  FILTER CudaCommon_REAL_OPERATOR_INSTANTIATIONS_TYPES
  INCLUDE
  REGEX "^(float|double)$"
)

set(CudaCommon_REAL_OPERATOR_INSTANTIATION_DEFINITIONS "")
foreach(type IN LISTS CudaCommon_REAL_OPERATOR_INSTANTIATIONS_TYPES)
  if(CudaCommon_REAL_OPERATOR_INSTANTIATION_DEFINITIONS)
    string(APPEND CudaCommon_REAL_OPERATOR_INSTANTIATION_DEFINITIONS " \\\n  ")
  endif()
  string(
    APPEND
    CudaCommon_REAL_OPERATOR_INSTANTIATION_DEFINITIONS
    "template decltype(OP<${type}>) CudaCommon_EXPORT OP<${type}>;"
  )
endforeach()

# Propagate cmake options in a header file at the same location as CudaCommonExport.h
if(ITK_SOURCE_DIR)
  set(
    _configuration_header_file
    "${ITKCommon_BINARY_DIR}/itkCudaCommonConfiguration.h"
  )
else()
  set(
    _configuration_header_file
    "${CudaCommon_BINARY_DIR}/include/itkCudaCommonConfiguration.h"
  )
endif()
configure_file(
  ${CudaCommon_SOURCE_DIR}/itkCudaCommonConfiguration.h.in
  ${_configuration_header_file}
)

set(
  CudaCommon_VERSION_STRINGS
  "
# CudaCommon version numbers
set(CudaCommon_VERSION_MAJOR ${CudaCommon_VERSION_MAJOR})
set(CudaCommon_VERSION_MINOR ${CudaCommon_VERSION_MINOR})
set(CudaCommon_VERSION_PATCH ${CudaCommon_VERSION_PATCH})
set(CudaCommon_VERSION_STRING ${CudaCommon_VERSION_STRING})
"
)

set(
  CudaCommon_EXPORT_CODE_BUILD
  "
# Pass source dir to allow other modules, e.g. RTK, to use CudaImage.i.init and CudaImage.i.in
set(CudaCommon_SOURCE_DIR ${CudaCommon_SOURCE_DIR})

${CudaCommon_VERSION_STRINGS}
"
)

set(
  CudaCommon_EXPORT_CODE_INSTALL
  "
find_package(CUDAToolkit REQUIRED)

${CudaCommon_VERSION_STRINGS}
"
)

if(NOT ITK_SOURCE_DIR)
  if(NOT EXISTS ${ITK_CMAKE_DIR}/ITKModuleMacros.cmake)
    message(
      FATAL_ERROR
      "Modules can only be built against an ITK build tree; they cannot be built against an ITK install tree."
    )
  endif()
  include(ITKModuleExternal)
else()
  set(ITK_DIR ${CMAKE_BINARY_DIR})
  itk_module_impl()
endif()

# Install configuration file
install(
  FILES
    ${_configuration_header_file}
  DESTINATION ${CudaCommon_INSTALL_INCLUDE_DIR}
)

if(NOT ITK_SOURCE_DIR)
  # Install the CMake targets export file so that find_package(ITK
  # COMPONENTS CudaCommon) can find the CudaCommon imported targets.
  install(
    EXPORT ${${itk-module}-targets}
    DESTINATION "${ITK_INSTALL_PACKAGE_DIR}/Modules/Targets"
    COMPONENT Development
  )
endif()
