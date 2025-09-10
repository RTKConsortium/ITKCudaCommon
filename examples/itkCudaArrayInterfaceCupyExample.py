import itk
import cupy as cp

def main():
    H, W = 16, 20
    GPUImageType = itk.CudaImage[itk.F, 2]
    img = GPUImageType.New()

    size = itk.Size[2]()
    size[0] = W  # x
    size[1] = H  # y
    region = itk.ImageRegion[2]()
    region.SetSize(size)
    img.SetBufferedRegion(region)
    img.SetSpacing([0.7, 0.7])
    img.SetOrigin([10.0, 20.0])
    img.FillBuffer(50)

    # convert to a CuPy array
    a_view = cp.asarray(img)

    # Do some operations with CuPy
    a_view += 2
    a_view *= 3

    # Back to ITK CudaImage
    img2 = itk.cuda_image_from_cuda_array(a_view)


if __name__ == "__main__":
    main()
