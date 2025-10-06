# CUDA array Interface in ITK

The CUDA array interface allows seamless interoperability between ITK's `CudaImage` and GPU arrays from libraries like [CuPy](https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html) and [PyTorch](https://docs.pytorch.org/docs/stable/tensors.html). This enables zero-copy data sharing, efficient GPU operations, and integration with deep learning pipelines.

## Overview

ITK's `CudaImage` implements the [`__cuda_array_interface__`](https://numba.pydata.org/numba-doc/0.43.0/cuda/cuda_array_interface.html) protocol, allowing it to be used directly as a [CuPy array](https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html) or [PyTorch tensor](https://docs.pytorch.org/docs/stable/tensors.html) without copying data to/from the GPU.

## Key Functions

- `itk.cuda_image_from_cuda_array(array)`\
Creates an ITK `CudaImage` from a CUDA array (CuPy ndarray or PyTorch tensor).
  - **Parameters**: `array` - A CuPy or PyTorch array on GPU.
  - **Returns**: A `CudaImage` with the same data (shared memory).
- `cp.asarray(cuda_image)` or `torch.as_tensor(cuda_image, device="cuda")`\
Creates a zero-copy view of the `CudaImage` as a CuPy array or PyTorch tensor.
  - **Parameters**: `cuda_image` - An ITK `CudaImage`.
  - **Returns**: A view of the GPU data (no copy).

## Examples

- [CuPy: `itkCudaArrayInterfaceCupyExample.py`](https://raw.githubusercontent.com/RTKConsortium/ITKCudaCommon/refs/heads/main/examples/itkCudaArrayInterfaceCupyExample.py)

```{literalinclude} examples/itkCudaArrayInterfaceCupyExample.py
```

- [Torch: `itkCudaArrayInterfaceTorchExample.py`](https://raw.githubusercontent.com/RTKConsortium/ITKCudaCommon/refs/heads/main/examples/itkCudaArrayInterfaceTorchExample.py)

```{literalinclude} examples/itkCudaArrayInterfaceCupyExample.py
```
