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
