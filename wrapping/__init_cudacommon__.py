import sys
import importlib

itk_module = sys.modules['itk']
cuda_submodules = ['itk.itkCudaImageFromImage']

for mod_name in cuda_submodules:
  mod = importlib.import_module(mod_name)
  for a in dir(mod):
    if a[0] != '_':
      setattr(itk_module, a, getattr(mod, a))
