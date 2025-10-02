#-----------------------------------------------------------------------------
# Find ITK to include ITK_USE_FILE and correctly configure dynamic libraries.
if(NOT ITK_DIR)
  set(ITK_DIR ${ITK_BINARY_DIR}/CMakeTmp)
endif()
find_package(ITK REQUIRED COMPONENTS ${ITK_MODULE_CudaCommon_DEPENDS})
include(${ITK_USE_FILE})

itk_module_test()

set(
  CudaCommonTests
  itkCudaSquareImageFilterTest.cxx
  itkCudaImageFromImageFilterTest.cxx
  itkCudaImageToImageFilterTest.cxx
  itkCudaMemoryProbeTest.cxx
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

if(ITK_WRAP_PYTHON)
  itk_python_add_test(NAME itkCudaImageFromImagePythonTest
    COMMAND itkCudaImageFromImageTest.py
  )
endif()
