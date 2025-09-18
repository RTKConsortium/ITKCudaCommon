import itk

# import importlib
# CudaImagePython = importlib.import_module('itk.itkCudaImagePython')


# Convert an object providing __cuda_array_interface__ to an itkCudaImage.
def cuda_image_from_cuda_array(a, keep_axes=False):
    desc = a.__cuda_array_interface__
    shape = list(desc["shape"])
    ptr = desc["data"][0]
    typestr = desc["typestr"]

    dim = len(shape)
    nb_comp = 1
    if dim > 3:
        nb_comp = shape[0]
        dim = dim - 1
        shape = shape[1:]

    if keep_axes == False:
        shape.reverse()

    size = itk.Size[dim](shape)

    # Force itk lazy loading to import itk.CudaCommon.get_itk_type
    itk.CudaImage

    pixelType = getattr(itk, itk.CudaCommon.get_itk_type(typestr))
    if nb_comp > 1:
        pixelType = itk.Vector[pixelType, nb_comp]

    GPUImageType = itk.CudaImage[pixelType, dim]
    cuda_img = GPUImageType.New()
    cuda_img.SetRegions(size)

    cuda_img.GetCudaDataManager().SetGPUBufferPointer(ptr)
    cuda_img.GetCudaDataManager().SetCPUDirtyFlag(True)
    cuda_img.GetCudaDataManager().SetGPUDirtyFlag(False)

    return cuda_img
