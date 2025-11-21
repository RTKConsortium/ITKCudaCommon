itk_wrap_module(CudaCommon)

set(
  WRAPPER_SUBMODULE_ORDER
  itkCudaDataManager
  itkCudaImage
  itkCudaImageDataManager
  itkCudaImageFromImageFilter
  itkCudaImageToImageFilter
  itkCudaInPlaceImageFilter
  itkCudaSquareImageFilter
  itkImageSourceCudaCommon
  itkImageToImageFilterCudaCommon
  itkInPlaceImageFilterCudaCommon
)

itk_auto_load_submodules()
itk_end_wrap_module()

wrap_itk_python_bindings_install(/itk "CudaCommon"
  __init_cudacommon__.py
  ${CudaCommon_SOURCE_DIR}/wrapping/itkCudaImageFromCudaArray.py
  ${CudaCommon_SOURCE_DIR}/wrapping/itkCudaImageFromImage.py
)
