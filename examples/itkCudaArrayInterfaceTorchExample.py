import itk
import torch

GPUImageType = itk.CudaImage[itk.F, 2]
img = GPUImageType.New()

img.SetSpacing([0.7, 0.7])
img.SetOrigin([10.0, 20.0])
img.SetRegions((32, 48))
img.Allocate()
img.FillBuffer(50)

# Convert to a zero-copy Torch GPU tensor view with DLPack. The tensor is
# created on the CUDA device used by the image, which is useful in multi-GPU
# workflows.
t_view = torch.from_dlpack(img)

# Do some operations with torch
t_view.mul_(2).add_(3)

# Convert back to a new ITK CudaImage
img2 = itk.cuda_image_from_cuda_array(t_view)
