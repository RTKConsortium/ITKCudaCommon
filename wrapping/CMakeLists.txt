itk_wrap_module(CudaCommon)
itk_auto_load_submodules()
itk_end_wrap_module()

wrap_itk_python_bindings_install(/itk "CudaCommon"
  __init_cudacommon__.py
  ${CudaCommon_SOURCE_DIR}/wrapping/itkCudaImageFromImage.py)

