# the top-level README is used for describing this module, just
# re-used it for documentation here
get_filename_component(MY_CURRENT_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

# itk_module() defines the module dependencies in CudaCommon
# CudaCommon depends on ITKCommon
# The testing module in CudaCommon depends on ITKTestKernel
# (besides CudaCommon and ITKCore)
# By convention those modules outside of ITK are not prefixed with
# ITK.

# define the dependencies of the include module and the tests
itk_module(CudaCommon
  DEPENDS
    ITKCommon
    ITKImageIntensity
  TEST_DEPENDS
    ITKTestKernel
  DESCRIPTION
  "ITKCudaCommon is an ITK module for processing images with Cuda, including the itk::CudaImage class and base classes for processing objects of this type, itk::CudaImageToImageFilter and itk::CudaInPlaceImageFilter. ITKCudaCommon was originally based on the GPUCommon module for OpenCL."
  EXCLUDE_FROM_DEFAULT
  ENABLE_SHARED
)
