itk_module_test()

set(ITKCudaCommon-Tests
 itkCudaSquareImageTest.cxx
 )

CreateTestDriver(ITKCudaCommon "${ITKCudaCommon-Test_LIBRARIES}" "${ITKCudaCommon-Tests}")

itk_add_test(NAME itkCudaSquareImageTest COMMAND ITKCudaCommonTestDriver itkCudaSquareImageTest)
