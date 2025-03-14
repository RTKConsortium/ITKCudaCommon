import itk
import sys

# Make the function available in the module's namespace
itk_module = sys.modules['itk']

# Convert an ITK image to a CUDA image if CUDA is available.
def cuda_image_from_image(img):
    if hasattr(itk, 'CudaImage'):
        img.Update()  # Ensure the image is up to date
        cuda_img = itk.CudaImage[itk.itkCType.GetCTypeForDType(img.dtype), img.ndim].New()
        cuda_img.SetPixelContainer(img.GetPixelContainer())
        cuda_img.CopyInformation(img)
        cuda_img.SetBufferedRegion(img.GetBufferedRegion())
        cuda_img.SetRequestedRegion(img.GetRequestedRegion())
        return cuda_img
    return img  # Return the original image if CUDA is not available

# Attach the function directly to the itk module
setattr(itk_module, 'cuda_image_from_image', cuda_image_from_image)