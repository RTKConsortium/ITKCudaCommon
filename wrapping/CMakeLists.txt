macro(cuda_wrap_image_filter_combinations from_types to_types prefix)
  set(parent ${ARGN})
  foreach(d ${ITK_WRAP_IMAGE_DIMS})
    foreach(from ${from_types})
      foreach(to ${to_types})
        if(parent)
          itk_wrap_template("${prefix}CI${ITKM_${from}}${d}CI${ITKM_${to}}${d}" "itk::CudaImage<${ITKT_${from}}, ${d}>, itk::CudaImage<${ITKT_${to}}, ${d}>, ${parent}<itk::CudaImage<${ITKT_${from}}, ${d}>, itk::CudaImage<${ITKT_${to}}, ${d}> >")
        else()
          itk_wrap_template("${prefix}CI${ITKM_${from}}${d}CI${ITKM_${to}}${d}" "itk::CudaImage<${ITKT_${from}}, ${d}>, itk::CudaImage<${ITKT_${to}}, ${d}>")
        endif()
      endforeach()
    endforeach()
  endforeach()
endmacro()

macro(cuda_wrap_vector_combinations vector_types prefix)
  set(parent ${ARGN})
  foreach(c ${ITK_WRAP_VECTOR_COMPONENTS})
    foreach(d ${ITK_WRAP_IMAGE_DIMS})
      foreach(vt ${vector_types})
        if(parent)
          itk_wrap_template("CI${ITKM_${vt}${c}}${d}CI${ITKM_${vt}${c}}${d}${prefix}" "itk::CudaImage<${ITKT_${vt}${c}}, ${d}>, itk::CudaImage<${ITKT_${vt}${c}}, ${d}>, ${parent}<itk::CudaImage<${ITKT_${vt}${c}}, ${d}>, itk::CudaImage<${ITKT_${vt}${c}}, ${d}> >")
        else()
          itk_wrap_template("CI${ITKM_${vt}${c}}${d}CI${ITKM_${vt}${c}}${d}${prefix}" "itk::CudaImage<${ITKT_${vt}${c}}, ${d}>, itk::CudaImage<${ITKT_${vt}${c}}, ${d}>")
        endif()
      endforeach()
    endforeach()
  endforeach()
endmacro()

itk_wrap_module(CudaCommon)

set(
  WRAPPER_SUBMODULE_ORDER
  itkCudaDataManager
  itkCudaImage
  itkCudaImageDataManager
  itkInPlaceImageFilterCudaCommon
  itkImageSourceCudaCommon
  itkImageToImageFilterCudaCommon
  itkCudaImageToImageFilter
  itkCudaInPlaceImageFilter
  itkCudaImageFromImageFilter
  itkCudaSquareImageFilter
)

itk_auto_load_submodules()
itk_end_wrap_module()

wrap_itk_python_bindings_install(/itk "CudaCommon"
  __init_cudacommon__.py
  ${CudaCommon_SOURCE_DIR}/wrapping/itkCudaImageFromCudaArray.py
  ${CudaCommon_SOURCE_DIR}/wrapping/itkCudaImageFromImage.py
)
