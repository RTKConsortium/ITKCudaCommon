import torch
import itk

def main():
    H, W = 32, 48
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

    # Convert to a Torch GPU tensor view
    t_view = torch.as_tensor(img, device="cuda")

    # Do some operations with torch
    t_view.mul_(2).add_(3)

    # Convert back to a new ITK CudaImage
    img2 = itk.cuda_image_from_cuda_array(t_view)


if __name__ == "__main__":
    main()