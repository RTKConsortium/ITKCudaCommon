import sys
import importlib


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


# Expose the global default CUDA device selection functions at the
# top-level `itk` namespace. The imports are deferred until call time
# because the SWIG sub-module is not fully usable during module init.
def set_default_cuda_device(device):
    from itk.itkCudaUtilPython import set_default_cuda_device as _set

    return _set(device)


def get_default_cuda_device():
    from itk.itkCudaUtilPython import get_default_cuda_device as _get

    return _get()


itk_module.set_default_cuda_device = set_default_cuda_device
itk_module.get_default_cuda_device = get_default_cuda_device
