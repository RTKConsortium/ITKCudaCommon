# CUDA Array Interface and DLPack in ITK

The CUDA array interface and DLPack allow seamless interoperability between ITK's `CudaImage` and GPU arrays from libraries like [CuPy](https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html) and [PyTorch](https://docs.pytorch.org/docs/stable/tensors.html). This enables zero-copy data sharing, efficient GPU operations, and integration with deep learning pipelines.

## Overview

ITK's `CudaImage` implements both the [`__cuda_array_interface__`](https://numba.pydata.org/numba-doc/0.43.0/cuda/cuda_array_interface.html) protocol and the [DLPack](https://dmlc.github.io/dlpack/latest/) protocol. Both provide zero-copy views of the image's GPU memory.

## Key Functions

- `itk.cuda_image_from_cuda_array(array)`\
Creates an ITK `CudaImage` from a CUDA array (CuPy ndarray or PyTorch tensor).
  - **Parameters**: `array` - A CuPy or PyTorch array on GPU.
  - **Returns**: A `CudaImage` with the same data (shared memory).
- `cp.asarray(cuda_image)` or `torch.as_tensor(cuda_image, device="cuda")`\
Creates a zero-copy view of the `CudaImage` as a CuPy array or PyTorch tensor using the CUDA array interface.
  - **Parameters**: `cuda_image` - An ITK `CudaImage`.
  - **Returns**: A view of the GPU data (no copy).
- `cp.from_dlpack(cuda_image)` or `torch.from_dlpack(cuda_image)`\
Creates a zero-copy view using DLPack.
  - **Parameters**: `cuda_image` - An ITK `CudaImage`.
  - **Returns**: A view of the GPU data (no copy) on the image's CUDA device.

## DLPack

DLPack is the recommended interoperability path when working with multiple GPUs. Unlike the CUDA array interface, DLPack includes device information through `__dlpack_device__`, so CuPy and PyTorch create the view on the same device as the `CudaImage`:

```python
import itk
import cupy as cp
import torch

image = itk.CudaImage[itk.F, 2].New()
image.SetRegions((16, 32))
image.Allocate()
image.FillBuffer(1.0)

cupy_view = cp.from_dlpack(image)
torch_view = torch.from_dlpack(image)
```

DLPack and the CUDA array interface are separate protocols. DLPack is used only through `from_dlpack`, while `cp.asarray(image)` and `torch.as_tensor(image, device="cuda")` continue to use the CUDA array interface.

While a DLPack view exists, automatic release of the image's dirty GPU buffer is disabled. Reading or using the image through ITK does not invalidate the view. Resizing the image or explicitly replacing or freeing its GPU buffer while a DLPack view exists is not supported and can invalidate the view; release the view first. Avoid modifying the image concurrently with operations on the DLPack view.

## Examples

- [CuPy: `itkCudaArrayInterfaceCupyExample.py`](https://raw.githubusercontent.com/RTKConsortium/ITKCudaCommon/refs/heads/main/examples/itkCudaArrayInterfaceCupyExample.py)

```{literalinclude} examples/itkCudaArrayInterfaceCupyExample.py
```

- [Torch: `itkCudaArrayInterfaceTorchExample.py`](https://raw.githubusercontent.com/RTKConsortium/ITKCudaCommon/refs/heads/main/examples/itkCudaArrayInterfaceTorchExample.py)

```{literalinclude} examples/itkCudaArrayInterfaceTorchExample.py
```
