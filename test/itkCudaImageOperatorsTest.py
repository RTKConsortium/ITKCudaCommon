import itk
import numpy as np


def main():
    image_type = itk.CudaImage[itk.F, 2]
    image = image_type.New()
    image.SetRegions([17, 19])
    image.Allocate()
    image.FillBuffer(2.0)

    reference = itk.SquareImageFilter[image_type, image_type].New()
    reference.SetInput(image)
    reference.InPlaceOff()
    reference.Update()

    operator = itk.CudaSquareImageFilter[image_type].New()
    operator.SetInput(image)
    operator.InPlaceOff()
    operator.GPUEnabledOn()
    operator.Update()

    cpu_image_type = itk.Image[itk.F, 2]
    np.testing.assert_array_equal(
        itk.array_from_image(operator.GetOutput(), ttype=cpu_image_type),
        itk.array_from_image(reference.GetOutput(), ttype=cpu_image_type),
    )


if __name__ == "__main__":
    main()
