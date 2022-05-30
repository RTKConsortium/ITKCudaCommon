set(CudaCommon_SRCS
  itkCudaContextManager.cxx
  itkCudaDataManager.cxx
  itkCudaUtil.cxx
  itkCudaMemoryProbe.cxx
  )

set(CudaCommon_Kernels
  itkCudaSquareImage.cu
  )

itk_module_add_library(CudaCommon ${CudaCommon_SRCS} ${CudaCommon_Kernels})
target_link_libraries(CudaCommon PUBLIC CUDA::cudart CUDA::cuda_driver)
target_include_directories(CudaCommon PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
set_property(TARGET CudaCommon PROPERTY CUDA_STANDARD ${CMAKE_CXX_STANDARD})
