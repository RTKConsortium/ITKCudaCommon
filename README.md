ITKCudaCommon
=================================

[![GitHub release](https://img.shields.io/github/release/RTKConsortium/ITKCudaCommon.svg)](https://github.com/RTKConsortium/ITKCudaCommon/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/itk-cudacommon-cuda116.svg)](https://pypi.python.org/pypi/itk-cudacommon-cuda116)
[![Wheels](https://img.shields.io/pypi/wheel/itk-cudacommon-cuda116.svg)](https://pypi.org/project/itk-cudacommon-cuda116)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/RTKConsortium/ITKCudaCommon/blob/main/LICENSE)
[![][gha-img]][gha-link]

[gha-img]: https://github.com/RTKConsortium/ITKCudaCommon/actions/workflows/build-test-package.yml/badge.svg
[gha-link]: https://github.com/RTKConsortium/ITKCudaCommon/actions/workflows/build-test-package.yml

Overview
--------

[ITKCudaCommon](https://github.com/RTKConsortium/ITKCudaCommon) is an [ITK](https://www.itk.org) module for processing images with Cuda, including the `itk::CudaImage` class and base classes for processing objects of this type, `itk::CudaImageToImageFilter` and `itk::CudaInPlaceImageFilter`. ITKCudaCommon was originally based on the [`GPUCommon` module](https://github.com/InsightSoftwareConsortium/ITK/tree/main/Modules/Core/GPUCommon) for OpenCL.

What is a CudaImage?
--------------------

An `itk::CudaImage` is an `itk::Image` (by inheritance) with a new member, `m_DataManager` of type `itk::CudaDataManager`. The role of this member is to manage a GPU memory buffer with the same size as the `itk::Image::m_Buffer`. Data is transferred from one buffer to the other (`m_GPUBuffer` and `m_CPUBuffer`) only when necessary, i.e., when accessing one of the two buffers and the other has been accessed more recently and therefore marked as "dirty" (using the members `m_IsGPUBufferDirty` and `m_IsCPUBufferDirty`).

The Python `CudaImage` wrapping exposes a [`__cuda_array_interface__`](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) for zero-copy views to other packages such as `PyTorch` or `CuPy`. See the [CUDA array interface documentation](./cuda_array_interface.md) for more information.

Selecting the GPU device
------------------------

By default, the GPU used by all freshly created `itk::CudaImage` (or `itk::CudaDataManager`) objects is selected automatically as the device with the maximum FLOPS. In a multi-GPU system you may want to choose a specific device instead.

Call this from C++ **before** creating any `CudaImage`:

```c++
// Use GPU number 2 for all subsequent images.
itk::SetDefaultCudaDevice(2);

// Reset to automatic selection (max FLOPS device).
itk::SetDefaultCudaDevice(-1);
```

The equivalent is available from Python:

```python
import itk

# Use GPU number 2 for all subsequently created CudaImages.
itk.set_default_cuda_device(2)

# Reset to automatic selection.
itk.set_default_cuda_device(-1)
```

Setting a device index outside the valid range throws an `itk::ExceptionObject`. The selection can also be controlled with the `ITK_CUDA_DEFAULT_DEVICE` environment variable.

What is a CudaImageToImageFilter?
---------------------------------

An `itk::CudaImageToImageFilter` and its in place child, `itk::CudaInPlaceImageFilter`, are ITK filters which can be integrated in an ITK pipeline to process an `itk::CudaImage`. They are CUDA equivalent to the CPU classes `itk::ImageToImageFilter` and `itk::InPlaceImageFilter` in [ITK's Common module](https://github.com/InsightSoftwareConsortium/ITK/tree/main/Modules/Core/Common). See the [ITK software guide](https://itk.org/ITKSoftwareGuide/html/) for more information. Their `GPUGenerateData` member function must process the output `CudaImage` objects.
