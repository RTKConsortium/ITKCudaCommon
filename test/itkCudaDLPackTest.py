import gc

import numpy as np
import pytest

import itk

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None


def create_cuda_image(size_list, fill_value, pixel_type=itk.F):
    Dim = len(size_list)
    GPUImageType = itk.CudaImage[pixel_type, Dim]
    img = GPUImageType.New()
    size = itk.Size[Dim]()
    for i in range(Dim):
        size[i] = size_list[i]
    img.SetRegions(size)
    img.Allocate()
    img.FillBuffer(fill_value)
    return img


def image_to_array(img):
    _, params = itk.template(img)
    PixelType, Dim = params
    CPUImageT = itk.Image[PixelType, Dim]
    return itk.GetArrayFromImage(img, ttype=CPUImageT)


def test_dlpack_cupy_2d():
    if cp is None:
        pytest.skip("CuPy is not installed")

    H, W = 16, 8
    img = create_cuda_image([W, H], 10)
    dm = img.GetCudaDataManager()

    assert img.__dlpack_device__() == (2, dm.GetDevice())

    a = cp.from_dlpack(img)
    assert a.shape == (H, W)
    assert a.dtype == cp.float32
    assert a.device.id == dm.GetDevice()
    assert a.data.ptr == int(dm.GetGPUBufferPointer())

    a += 5
    np.testing.assert_allclose(cp.asnumpy(a), np.full((H, W), 15, dtype=np.float32))
    del a

    arr = image_to_array(img)
    np.testing.assert_allclose(arr, np.full((H, W), 15, dtype=arr.dtype))


def test_dlpack_cupy_3d():
    if cp is None:
        pytest.skip("CuPy is not installed")

    Z, Y, X = 4, 3, 5
    img = create_cuda_image([X, Y, Z], 10)

    a = cp.from_dlpack(img)
    assert a.shape == (Z, Y, X)
    assert a.dtype == cp.float32

    a += 5
    np.testing.assert_allclose(cp.asnumpy(a), np.full((Z, Y, X), 15, dtype=np.float32))


def test_dlpack_cupy_vector():
    if cp is None:
        pytest.skip("CuPy is not installed")

    H, W = 3, 4
    pixel_type = itk.Vector[itk.F, 3]
    value = itk.Vector[itk.F, 3]()
    value[0] = 1
    value[1] = 2
    value[2] = 3
    img = create_cuda_image([W, H], value, pixel_type)

    a = cp.from_dlpack(img)
    assert a.shape == (3, H, W)
    np.testing.assert_allclose(cp.asnumpy(a[:, 0, 0]), [1, 2, 3])

    a += 10
    np.testing.assert_allclose(cp.asnumpy(a[:, 0, 0]), [11, 12, 13])


def test_dlpack_view_survives_itk_read():
    if cp is None:
        pytest.skip("CuPy is not installed")

    H, W = 16, 8
    img = create_cuda_image([W, H], 10)
    a = cp.from_dlpack(img)
    a += 5

    # Reading the image through ITK must not free the GPU buffer used by the view.
    arr = image_to_array(img)
    np.testing.assert_allclose(arr, np.full((H, W), 15, dtype=arr.dtype))
    np.testing.assert_allclose(cp.asnumpy(a), np.full((H, W), 15, dtype=np.float32))


def test_dlpack_view_outlives_image():
    if cp is None:
        pytest.skip("CuPy is not installed")

    img = create_cuda_image([4, 5], 10)
    a = cp.from_dlpack(img)
    del img
    gc.collect()

    np.testing.assert_allclose(cp.asnumpy(a), np.full((5, 4), 10, dtype=np.float32))
    del a
    gc.collect()


def test_dlpack_release_flag_is_refcounted():
    if cp is None:
        pytest.skip("CuPy is not installed")

    img = create_cuda_image([4, 5], 10)
    dm = img.GetCudaDataManager()
    assert dm.GetReleaseDirtyGPUBuffer()

    a = cp.from_dlpack(img)
    b = cp.from_dlpack(img)
    assert not dm.GetReleaseDirtyGPUBuffer()

    del a
    gc.collect()
    assert not dm.GetReleaseDirtyGPUBuffer()

    del b
    gc.collect()
    assert dm.GetReleaseDirtyGPUBuffer()


def test_dlpack_torch_2d():
    if torch is None:
        pytest.skip("PyTorch is not installed")

    H, W = 10, 12
    img = create_cuda_image([W, H], 10)
    dm = img.GetCudaDataManager()

    t = torch.from_dlpack(img)
    assert tuple(t.shape) == (H, W)
    assert t.dtype == torch.float32
    assert t.device.index == dm.GetDevice()

    t.add_(5)
    np.testing.assert_allclose(t.cpu().numpy(), np.full((H, W), 15, dtype=np.float32))
    del t

    arr = image_to_array(img)
    np.testing.assert_allclose(arr, np.full((H, W), 15, dtype=arr.dtype))


def test_dlpack_copy_is_rejected():
    img = create_cuda_image([4, 5], 1)
    with pytest.raises(BufferError):
        img.__dlpack__(copy=True)


def test_dlpack_dl_device_mismatch_is_rejected():
    img = create_cuda_image([4, 5], 1)
    device = img.__dlpack_device__()[1]
    with pytest.raises(BufferError):
        img.__dlpack__(dl_device=(2, device + 1))


@pytest.mark.skipif(
    cp is None or cp.cuda.runtime.getDeviceCount() < 2,
    reason="DLPack multi-GPU test requires CuPy and at least two CUDA devices",
)
def test_dlpack_multi_gpu():
    previous_default = itk.get_default_cuda_device()
    try:
        itk.set_default_cuda_device(0)
        img0 = create_cuda_image([4, 5], 10)

        itk.set_default_cuda_device(1)
        img1 = create_cuda_image([4, 5], 20)

        a0 = cp.from_dlpack(img0)
        a1 = cp.from_dlpack(img1)
        try:
            assert a0.device.id == img0.GetCudaDataManager().GetDevice() == 0
            assert a1.device.id == img1.GetCudaDataManager().GetDevice() == 1
            np.testing.assert_allclose(
                cp.asnumpy(a0), np.full((5, 4), 10, dtype=np.float32)
            )
            np.testing.assert_allclose(
                cp.asnumpy(a1), np.full((5, 4), 20, dtype=np.float32)
            )
        finally:
            del a0, a1
    finally:
        itk.set_default_cuda_device(previous_default)
