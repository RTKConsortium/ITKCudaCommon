import itk

# Convert an ITK image to a CUDA image if CUDA is available.
def cuda_image_from_image(img):
  img.Update()  # Ensure the image is up to date
  cuda_img = itk.CudaImage[itk.itkCType.GetCTypeForDType(img.dtype), img.ndim].New()
  cuda_img.SetPixelContainer(img.GetPixelContainer())
  cuda_img.CopyInformation(img)
  cuda_img.SetBufferedRegion(img.GetBufferedRegion())
  cuda_img.SetRequestedRegion(img.GetRequestedRegion())
  return cuda_img
