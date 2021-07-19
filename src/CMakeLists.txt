set(ITKCudaCommon_SRCS
    itkCudaContextManager.cxx
    itkCudaDataManager.cxx
    itkCudaUtil.cxx
    itkCudaMemoryProbe.cxx
  )

itk_module_add_library(ITKCudaCommon ${ITKCudaCommon_SRCS})
target_link_libraries(ITKCudaCommon LINK_PUBLIC CUDA::cudart CUDA::cuda_driver)
target_include_directories(ITKCudaCommon PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_property(TARGET ITKCudaCommon PROPERTY CUDA_STANDARD ${CMAKE_CXX_STANDARD})
