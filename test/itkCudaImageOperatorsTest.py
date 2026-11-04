import itk
import numpy as np


def make_image(image_type, values):
    image = image_type.New()
    size = itk.Size[2]()
    size[0], size[1] = values.shape[1], values.shape[0]
    image.SetRegions(size)
    image.Allocate()
    index = itk.Index[2]()
    for (y, x), value in np.ndenumerate(values):
        index[0], index[1] = x, y
        image.SetPixel(index, float(value))
    return image


def check_image_quality(image, reference):
    region = image.GetBufferedRegion()
    reference_region = reference.GetBufferedRegion()
    assert tuple(region.GetIndex()) == tuple(reference_region.GetIndex())
    assert tuple(region.GetSize()) == tuple(reference_region.GetSize())

    cpu_image_type = itk.Image[itk.F, 2]
    actual = itk.array_from_image(image, ttype=cpu_image_type)
    expected = itk.array_from_image(reference, ttype=cpu_image_type)
    # Match itk::Math::AlmostEquals<float>, including its near-zero tolerance.
    with np.errstate(over="ignore", invalid="ignore"):
        close = (actual == expected) | (
            np.abs(actual - expected) <= np.float32(0.1 * np.finfo(np.float32).eps)
        )
    if np.any(~close):
        np.testing.assert_array_max_ulp(actual[~close], expected[~close], maxulp=4)
    zeros = (actual == 0) & (expected == 0)
    np.testing.assert_array_equal(
        np.signbit(actual[zeros]), np.signbit(expected[zeros])
    )


def check_unary_operator(
    cuda_filter_type, cpu_filter_type, input_offset=0.0, input_scale=0.25
):
    image_type = itk.template(cuda_filter_type)[1][0]
    pixels = np.arange(17 * 19).reshape(19, 17)
    values = np.float32(input_offset) + (pixels % 21 - 10).astype(
        np.float32
    ) * np.float32(input_scale)
    if cuda_filter_type in (
        itk.CudaAbsImageFilter[image_type],
        itk.CudaRoundImageFilter[image_type],
    ):
        values[0, 0] = -0.0
        values[0, 1] = np.nextafter(np.float32(0.5), np.float32(0.0))
    image = make_image(image_type, values)

    reference = cpu_filter_type.New()
    reference.SetInput(image)
    reference.InPlaceOff()
    reference.Update()

    operator = cuda_filter_type.New()
    operator.SetInput(image)
    operator.InPlaceOff()
    operator.GPUEnabledOn()
    operator.Update()

    region = operator.GetOutput().GetBufferedRegion()
    assert tuple(region.GetIndex()) == (0, 0)
    assert tuple(region.GetSize()) == (17, 19)
    print(f"Checking {operator.GetNameOfClass()} against {reference.GetNameOfClass()}")
    check_image_quality(operator.GetOutput(), reference.GetOutput())


def main():
    Image = itk.CudaImage[itk.F, 2]
    check_unary_operator(
        itk.CudaSquareImageFilter[Image], itk.SquareImageFilter[Image, Image]
    )
    check_unary_operator(
        itk.CudaSqrtImageFilter[Image],
        itk.SqrtImageFilter[Image, Image],
        input_offset=2.5,
    )
    check_unary_operator(
        itk.CudaAbsImageFilter[Image], itk.AbsImageFilter[Image, Image]
    )
    check_unary_operator(
        itk.CudaAcosImageFilter[Image],
        itk.AcosImageFilter[Image, Image],
        input_scale=0.1,
    )
    check_unary_operator(
        itk.CudaAsinImageFilter[Image],
        itk.AsinImageFilter[Image, Image],
        input_scale=0.1,
    )
    check_unary_operator(
        itk.CudaAtanImageFilter[Image], itk.AtanImageFilter[Image, Image]
    )
    check_unary_operator(
        itk.CudaBoundedReciprocalImageFilter[Image],
        itk.BoundedReciprocalImageFilter[Image, Image],
    )
    check_unary_operator(
        itk.CudaCosImageFilter[Image], itk.CosImageFilter[Image, Image]
    )
    check_unary_operator(
        itk.CudaExpImageFilter[Image], itk.ExpImageFilter[Image, Image]
    )
    check_unary_operator(
        itk.CudaLogImageFilter[Image],
        itk.LogImageFilter[Image, Image],
        input_offset=3.0,
    )
    check_unary_operator(
        itk.CudaLog10ImageFilter[Image],
        itk.Log10ImageFilter[Image, Image],
        input_offset=3.0,
    )
    check_unary_operator(
        itk.CudaRoundImageFilter[Image], itk.RoundImageFilter[Image, Image]
    )
    check_unary_operator(
        itk.CudaSinImageFilter[Image], itk.SinImageFilter[Image, Image]
    )
    check_unary_operator(
        itk.CudaTanImageFilter[Image], itk.TanImageFilter[Image, Image]
    )


if __name__ == "__main__":
    main()
