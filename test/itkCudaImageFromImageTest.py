import itk

def test_cuda_image_from_image():
    # Define image type (2D float Image)
    PixelType = itk.F
    Dimension = 2
    ImageType = itk.Image[PixelType, Dimension]

    # Set size and region
    size = itk.Size[Dimension]()
    size.Fill(256)

    region = itk.ImageRegion[Dimension]()
    region.SetSize(size)

    # Create and allocate the CUDA image
    image = ImageType.New()
    image.SetRegions(region)
    image.Allocate()

    # Fill image with a constant value
    fill_value = 2.0
    image.FillBuffer(fill_value)

    # Convert image
    cuda_img = itk.cuda_image_from_image(image)

    # Check results
    assert isinstance(cuda_img, itk.CudaImage), "Failed to convert to CudaImage"
