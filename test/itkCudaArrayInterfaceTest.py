import numpy as np
import itk
import cupy as cp
import torch


def create_cuda_image(size_list, fill_value=None):
    Dim = len(size_list)
    GPUImageType = itk.CudaImage[itk.F, Dim]
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


def test_cuda_array_interface_cupy_2d():
    H, W = 16, 8
    img = create_cuda_image([W, H], fill_value=10)
    a_view = cp.asarray(img)
    a_view += 5
    img2 = itk.cuda_image_from_cuda_array(a_view)
    assert np.allclose(img2.GetSpacing(), img.GetSpacing())
    assert np.allclose(img2.GetOrigin(), img.GetOrigin())
    assert np.allclose(img2.GetDirection(), img.GetDirection())
    arr2 = image_to_array(img2)
    np.testing.assert_allclose(arr2, np.full((H, W), 15, dtype=arr2.dtype))


def test_cuda_array_interface_cupy_3d():
    Z, Y, X = 4, 3, 5
    img = create_cuda_image([X, Y, Z], fill_value=10)
    a_view = cp.asarray(img)
    a_view += 5
    img2 = itk.cuda_image_from_cuda_array(a_view)
    assert np.allclose(img2.GetSpacing(), img.GetSpacing())
    assert np.allclose(img2.GetOrigin(), img.GetOrigin())
    assert np.allclose(img2.GetDirection(), img.GetDirection())
    arr2 = image_to_array(img2)
    np.testing.assert_allclose(arr2, np.full((Z, Y, X), 15, dtype=arr2.dtype))


def test_cuda_array_interface_torch_2d():
    H, W = 10, 12
    img = create_cuda_image([W, H], fill_value=10)
    t_view = torch.as_tensor(img, device="cuda")
    t_view.add_(5)
    img2 = itk.cuda_image_from_cuda_array(t_view)
    assert np.allclose(img2.GetSpacing(), img.GetSpacing())
    assert np.allclose(img2.GetOrigin(), img.GetOrigin())
    assert np.allclose(img2.GetDirection(), img.GetDirection())
    arr2 = image_to_array(img2)
    np.testing.assert_allclose(arr2, np.full((H, W), 15, dtype=arr2.dtype))


def test_cuda_array_interface_torch_3d():
    Z, Y, X = 2, 3, 4
    img = create_cuda_image([X, Y, Z], fill_value=10)
    t_view = torch.as_tensor(img, device="cuda")
    t_view.add_(5)
    img2 = itk.cuda_image_from_cuda_array(t_view)
    assert np.allclose(img2.GetSpacing(), img.GetSpacing())
    assert np.allclose(img2.GetOrigin(), img.GetOrigin())
    assert np.allclose(img2.GetDirection(), img.GetDirection())
    arr2 = image_to_array(img2)
    np.testing.assert_allclose(arr2, np.full((Z, Y, X), 15, dtype=arr2.dtype))


def test_cupy_getpixel():
    x_gpu = cp.array([[0, 1, 2], [3, 4, 5]], dtype=cp.float32, order="C")
    cuda_img = itk.cuda_image_from_cuda_array(x_gpu)

    pixelIndex = itk.Index[2]()
    pixelIndex[0] = 2  # x
    pixelIndex[1] = 1  # y

    val = cuda_img.GetPixel(pixelIndex)
    assert np.isclose(val, 5.0)


def test_torch_getpixel():
    t = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float32, device="cuda")
    cuda_img = itk.cuda_image_from_cuda_array(t)

    pixelIndex = itk.Index[2]()
    pixelIndex[0] = 2  # x
    pixelIndex[1] = 1  # y

    val = cuda_img.GetPixel(pixelIndex)
    assert np.isclose(val, 5.0)
