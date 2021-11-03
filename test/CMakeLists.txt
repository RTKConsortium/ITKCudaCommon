itk_module_test()

set(CudaCommonTests
 itkCudaSquareImageFilterTest.cxx
  )

CreateTestDriver(CudaCommon "${CudaCommon-Test_LIBRARIES}" "${CudaCommonTests}")

itk_add_test(NAME itkCudaSquareImageFilterTest
  COMMAND CudaCommonTestDriver
  itkCudaSquareImageFilterTest
  )
