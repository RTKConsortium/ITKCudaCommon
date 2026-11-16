set(
  CudaCommon_SRCS
  itkCudaDataManager.cxx
  itkCudaUtil.cxx
  itkCudaMemoryProbe.cxx
)

set(CudaCommon_Kernels itkCudaUnaryImageOperators.cu)

itk_module_add_library(CudaCommon ${CudaCommon_SRCS} ${CudaCommon_Kernels})

target_compile_options(
  CudaCommon
  PUBLIC
    "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>"
)
target_link_libraries(
  CudaCommon
  PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)
target_include_directories(CudaCommon PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
set_property(
  TARGET
    CudaCommon
  PROPERTY
    CUDA_STANDARD
      ${CMAKE_CXX_STANDARD}
)
