import sys
import importlib
import os


itk_module = sys.modules["itk"]

# Load the CMake-generated version and assign it to
# `itk.CudaCommon.__version__`.
cudacommon_version = importlib.import_module(
    "itk.cudacommonConfig"
).CUDACOMMON_GLOBAL_VERSION_STRING
cudacommon_module = getattr(itk_module, "CudaCommon")
setattr(cudacommon_module, "__version__", cudacommon_version)

cuda_submodules = ["itk.itkCudaImageFromImage", "itk.itkCudaImageFromCudaArray"]

for mod_name in cuda_submodules:
    mod = importlib.import_module(mod_name)
    for a in dir(mod):
        if a[0] != "_":
            setattr(itk_module, a, getattr(mod, a))


def set_default_cuda_device(device):
    """Set the default CUDA device used by all subsequently created CudaImages.

    Pass an integer device index (0, 1, ...) to select that GPU, or -1 to
    reset to automatic selection (the max FLOPS device). This is the same
    behavior as the C++ itk::SetDefaultCudaDevice function.
    """
    os.environ["ITK_CUDA_DEFAULT_DEVICE"] = str(int(device))


setattr(itk_module, "set_default_cuda_device", set_default_cuda_device)
