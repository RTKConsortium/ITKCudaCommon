#-----------------------------------------------------------------------------
# Find ITK to include ITK_USE_FILE and correctly configure dynamic libraries.
if(NOT ITK_DIR)
  set(ITK_DIR ${ITK_BINARY_DIR}/CMakeTmp)
endif()
find_package(ITK REQUIRED COMPONENTS ${ITK_MODULE_CudaCommon_DEPENDS})
include(itkVersion)
if("${ITK_VERSION_MAJOR}" VERSION_LESS "6")
  include(${ITK_USE_FILE})
endif()

itk_module_test()

set(
  CudaCommonTests
  itkCudaSquareImageFilterTest.cxx
  itkCudaImageFromImageFilterTest.cxx
  itkCudaImageToImageFilterTest.cxx
  itkCudaMemoryProbeTest.cxx
  itkCudaDataManagerDirtyFlagTest.cxx
  itkCudaDataManagerReleaseFreeTest.cxx
  itkCudaImageRegionGraftTest.cxx
)

createtestdriver(CudaCommon "${CudaCommon-Test_LIBRARIES}" "${CudaCommonTests}")

itk_add_test(NAME itkCudaSquareImageFilterTest
  COMMAND CudaCommonTestDriver
  itkCudaSquareImageFilterTest
)

itk_add_test(NAME itkCudaImageFromImageFilterTest
  COMMAND CudaCommonTestDriver
  itkCudaImageFromImageFilterTest
)

itk_add_test(NAME itkCudaImageToImageFilterTest
  COMMAND CudaCommonTestDriver
  itkCudaImageToImageFilterTest
)

itk_add_test(NAME itkCudaMemoryProbeTest
  COMMAND CudaCommonTestDriver
  itkCudaMemoryProbeTest
)

itk_add_test(NAME itkCudaDataManagerDirtyFlagTest
  COMMAND CudaCommonTestDriver
  itkCudaDataManagerDirtyFlagTest
)

itk_add_test(NAME itkCudaDataManagerReleaseFreeTest
  COMMAND CudaCommonTestDriver
  itkCudaDataManagerReleaseFreeTest
)

itk_add_test(NAME itkCudaImageRegionGraftTest
  COMMAND CudaCommonTestDriver
  itkCudaImageRegionGraftTest
)

if(ITK_WRAP_PYTHON)
  itk_python_add_test(NAME itkCudaImageFromImagePythonTest
    COMMAND itkCudaImageFromImageTest.py
  )
endif()
