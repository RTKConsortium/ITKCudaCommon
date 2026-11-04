itk_wrap_module(CudaCommon)

# Derive ITK wrapping aliases from the configured C++ pixel types.
set(operator_types "${CudaCommon_OPERATOR_INSTANTIATIONS_TYPES}")
list(
  TRANSFORM operator_types
  REPLACE
    "^signed "
    ""
)
set(operator_aliases "F;D;SI;UI;UL;SS;US;UC;SC;SL;SLL;ULL")
set(CudaCommon_WRAP_OPERATOR_TYPES "")
foreach(alias IN LISTS operator_aliases)
  string(REGEX REPLACE "^signed " "" type "${ITKT_${alias}}")
  if(type IN_LIST operator_types)
    list(APPEND CudaCommon_WRAP_OPERATOR_TYPES "${alias}")
  endif()
endforeach()

set(CudaCommon_WRAP_REAL_OPERATOR_TYPES "${CudaCommon_WRAP_OPERATOR_TYPES}")
list(FILTER CudaCommon_WRAP_REAL_OPERATOR_TYPES INCLUDE REGEX "^(F|D)$")

set(
  WRAPPER_SUBMODULE_ORDER
  itkCudaDataManager
  itkCudaImage
  itkCudaImageDataManager
  itkImageToImageFilterCudaCommon
  itkInPlaceImageFilterCudaCommon
  itkImageSourceCudaCommon
  itkUnaryGeneratorImageFilterCudaCommon
  itkAbsImageFilterCudaCommon
  itkAcosImageFilterCudaCommon
  itkAsinImageFilterCudaCommon
  itkAtanImageFilterCudaCommon
  itkBoundedReciprocalImageFilterCudaCommon
  itkCosImageFilterCudaCommon
  itkExpImageFilterCudaCommon
  itkLogImageFilterCudaCommon
  itkLog10ImageFilterCudaCommon
  itkRoundImageFilterCudaCommon
  itkSinImageFilterCudaCommon
  itkSqrtImageFilterCudaCommon
  itkSquareImageFilterCudaCommon
  itkTanImageFilterCudaCommon
  itkCudaImageToImageFilter
  itkCudaUnaryImageToImageFilter
  itkCudaInPlaceImageFilter
  itkCudaImageFromImageFilter
  itkCudaAbsImageFilter
  itkCudaAcosImageFilter
  itkCudaAsinImageFilter
  itkCudaAtanImageFilter
  itkCudaBoundedReciprocalImageFilter
  itkCudaCosImageFilter
  itkCudaExpImageFilter
  itkCudaLogImageFilter
  itkCudaLog10ImageFilter
  itkCudaRoundImageFilter
  itkCudaSinImageFilter
  itkCudaSqrtImageFilter
  itkCudaSquareImageFilter
  itkCudaTanImageFilter
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
