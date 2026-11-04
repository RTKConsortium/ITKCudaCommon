itk_wrap_module(CudaCommon)

# Scalar types supported by the CUDA operators.
set(CudaCommonOperatorTypes "UC;UL;${WRAP_ITK_SCALAR}")
list(
  REMOVE_ITEM
  CudaCommonOperatorTypes
  SC
  SL
  SLL
  ULL
)
list(REMOVE_DUPLICATES CudaCommonOperatorTypes)

set(
  WRAPPER_SUBMODULE_ORDER
  itkCudaDataManager
  itkCudaImage
  itkCudaImageDataManager
  itkImageToImageFilterCudaCommon
  itkInPlaceImageFilterCudaCommon
  itkImageSourceCudaCommon
  itkUnaryGeneratorImageFilterCudaCommon
  itkSquareImageFilterCudaCommon
  itkSqrtImageFilterCudaCommon
  itkCudaImageToImageFilter
  itkCudaUnaryImageToImageFilter
  itkCudaInPlaceImageFilter
  itkCudaImageFromImageFilter
  itkCudaSquareImageFilter
  itkCudaSqrtImageFilter
)

configure_file(
  ${CudaCommon_SOURCE_DIR}/wrapping/cudacommonConfig.py.in
  ${CudaCommon_BINARY_DIR}/Wrapping/Generators/Python/cudacommonConfig.py
)

itk_auto_load_submodules()
itk_end_wrap_module()

wrap_itk_python_bindings_install(/itk "CudaCommon"
  __init_cudacommon__.py
  ${CudaCommon_BINARY_DIR}/Wrapping/Generators/Python/cudacommonConfig.py
  ${CudaCommon_SOURCE_DIR}/wrapping/itkCudaImageFromCudaArray.py
  ${CudaCommon_SOURCE_DIR}/wrapping/itkCudaImageFromImage.py
)
