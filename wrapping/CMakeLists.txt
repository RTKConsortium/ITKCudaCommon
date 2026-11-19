itk_wrap_module(CudaCommon)

set(
  WRAPPER_SUBMODULE_ORDER
  itkCudaDataManager
  itkCudaImage
  itkCudaImageDataManager
  itkImageToImageFilterCudaCommon
  itkInPlaceImageFilterCudaCommon
  itkImageSourceCudaCommon
  itkUnaryGeneratorImageFilterCudaCommon
  itkBinaryGeneratorImageFilterCudaCommon
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
  itkThresholdImageFilterCudaCommon
  itkAddImageFilterCudaCommon
  itkSubtractImageFilterCudaCommon
  itkMultiplyImageFilterCudaCommon
  itkDivideImageFilterCudaCommon
  itkDivideOrZeroOutImageFilterCudaCommon
  itkWeightedAddImageFilterCudaCommon
  itkModulusImageFilterCudaCommon
  itkAndImageFilterCudaCommon
  itkOrImageFilterCudaCommon
  itkXorImageFilterCudaCommon
  itkMaximumImageFilterCudaCommon
  itkMinimumImageFilterCudaCommon
  itkPowImageFilterCudaCommon
  itkAtan2ImageFilterCudaCommon
  itkBinaryMagnitudeImageFilterCudaCommon
  itkConstrainedValueAdditionImageFilterCudaCommon
  itkConstrainedValueDifferenceImageFilterCudaCommon
  itkMaskImageFilterCudaCommon
  itkMaskNegatedImageFilterCudaCommon
  itkSquareImageFilterCudaCommon
  itkTanImageFilterCudaCommon
  itkCudaImageToImageFilter
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
  itkCudaThresholdImageFilter
  itkCudaAddImageFilter
  itkCudaSubtractImageFilter
  itkCudaMultiplyImageFilter
  itkCudaDivideImageFilter
  itkCudaDivideOrZeroOutImageFilter
  itkCudaWeightedAddImageFilter
  itkCudaModulusImageFilter
  itkCudaAndImageFilter
  itkCudaOrImageFilter
  itkCudaXorImageFilter
  itkCudaMaximumImageFilter
  itkCudaMinimumImageFilter
  itkCudaPowImageFilter
  itkCudaAtan2ImageFilter
  itkCudaBinaryMagnitudeImageFilter
  itkCudaConstrainedValueAdditionImageFilter
  itkCudaConstrainedValueDifferenceImageFilter
  itkCudaMaskImageFilter
  itkCudaMaskNegatedImageFilter
  itkCudaSquareImageFilter
  itkCudaTanImageFilter
)

# Mask pixel types of the mask filters, which are instantiated in
# src/itkCudaBinaryImageOperators.cu: the wrapped integer types among those
# instantiated, as in ITK, plus the image pixel type in each .wrap file.
set(CudaCommon_WRAP_MASK_TYPES)
foreach(m UC US SS)
  if(m IN_LIST WRAP_ITK_INT)
    list(APPEND CudaCommon_WRAP_MASK_TYPES ${m})
  endif()
endforeach()

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
