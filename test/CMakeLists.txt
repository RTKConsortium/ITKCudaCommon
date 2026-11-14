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
  itkCudaImageOperatorsTest.cxx
  itkCudaImageFromImageFilterTest.cxx
  itkCudaImageToImageFilterTest.cxx
  itkCudaMemoryProbeTest.cxx
  itkCudaDataManagerDirtyFlagTest.cxx
  itkCudaImageDirtyFlagTest.cxx
  itkCudaDataManagerReleaseFreeTest.cxx
  itkCudaImageRegionGraftTest.cxx
  itkCudaDeviceSelectionTest.cxx
)

createtestdriver(CudaCommon "${CudaCommon-Test_LIBRARIES}" "${CudaCommonTests}")

# Let the tests skip pixel types whose operators are not instantiated.
foreach(type IN LISTS CudaCommon_OPERATOR_INSTANTIATIONS_TYPES)
  string(REGEX REPLACE "^signed " "" type "${type}")
  string(MAKE_C_IDENTIFIER "${type}" type_id)
  target_compile_definitions(
    CudaCommonTestDriver
    PRIVATE
      "CudaCommon_TEST_${type_id}"
  )
endforeach()

itk_add_test(NAME itkCudaImageOperatorsTest
  COMMAND CudaCommonTestDriver
  itkCudaImageOperatorsTest
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

itk_add_test(NAME itkCudaImageDirtyFlagTest
  COMMAND CudaCommonTestDriver
  itkCudaImageDirtyFlagTest
)

itk_add_test(NAME itkCudaDataManagerReleaseFreeTest
  COMMAND CudaCommonTestDriver
  itkCudaDataManagerReleaseFreeTest
)

itk_add_test(NAME itkCudaImageRegionGraftTest
  COMMAND CudaCommonTestDriver
  itkCudaImageRegionGraftTest
)

itk_add_test(NAME itkCudaDeviceSelectionTest
  COMMAND CudaCommonTestDriver
  itkCudaDeviceSelectionTest
)

if(ITK_WRAP_PYTHON)
  itk_python_add_test(NAME itkCudaImageFromImagePythonTest
    COMMAND itkCudaImageFromImageTest.py
  )
  itk_python_add_test(NAME itkCudaImageOperatorsPythonTest
    COMMAND itkCudaImageOperatorsTest.py
  )
endif()
