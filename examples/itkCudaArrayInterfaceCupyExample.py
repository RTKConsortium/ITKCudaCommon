import itk
import cupy as cp

GPUImageType = itk.CudaImage[itk.F, 2]
img = GPUImageType.New()

img.SetSpacing([0.7, 0.7])
img.SetOrigin([10.0, 20.0])
img.SetRegions((16, 20))
img.Allocate()
img.FillBuffer(50)

# Create a zero-copy CuPy view with DLPack. The view is created on the
# CUDA device used by the image, which is useful in multi-GPU workflows.
a_view = cp.from_dlpack(img)

# Do some operations with CuPy
a_view += 2
a_view *= 3

# Back to ITK CudaImage
img2 = itk.cuda_image_from_cuda_array(a_view)
