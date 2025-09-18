import itk


def cuda_image_from_image(image):
    """Convert a CPU itk.Image instance to an itk.CudaImage using CudaImageFromImageFilter."""
    InputImageT = image.__class__
    FilterT = itk.CudaImageFromImageFilter[InputImageT]
    flt = FilterT.New()
    flt.SetInput(image)
    flt.Update()
    return flt.GetOutput()
