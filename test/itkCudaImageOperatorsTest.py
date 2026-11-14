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


def check_image_quality(image, reference, pixel_type):
    region = image.GetBufferedRegion()
    reference_region = reference.GetBufferedRegion()
    assert tuple(region.GetIndex()) == tuple(reference_region.GetIndex())
    assert tuple(region.GetSize()) == tuple(reference_region.GetSize())

    float_type = np.float32 if pixel_type == itk.F else np.float64
    cpu_image_type = itk.Image[pixel_type, 2]
    actual = np.asarray(
        itk.array_from_image(image, ttype=cpu_image_type), dtype=float_type
    )
    expected = np.asarray(
        itk.array_from_image(reference, ttype=cpu_image_type), dtype=float_type
    )
    # Match itk::Math::AlmostEquals<PIXEL>, including its near-zero tolerance.
    with np.errstate(over="ignore", invalid="ignore"):
        close = (actual == expected) | (
            np.abs(actual - expected) <= float_type(0.1 * np.finfo(float_type).eps)
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
    pixel_type = itk.template(image_type)[1][0]
    float_type = np.float32 if pixel_type == itk.F else np.float64
    pixels = np.arange(17 * 19).reshape(19, 17)
    values = float_type(input_offset) + (pixels % 21 - 10).astype(
        float_type
    ) * float_type(input_scale)
    if cuda_filter_type in (
        itk.CudaAbsImageFilter[image_type],
        itk.CudaRoundImageFilter[image_type],
    ):
        values[0, 0] = float_type(-0.0)
        values[0, 1] = np.nextafter(float_type(0.5), float_type(0.0))
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
    check_image_quality(operator.GetOutput(), reference.GetOutput(), pixel_type)


def check_operators(image_type):
    check_unary_operator(
        itk.CudaSquareImageFilter[image_type],
        itk.SquareImageFilter[image_type, image_type],
    )
    check_unary_operator(
        itk.CudaSqrtImageFilter[image_type],
        itk.SqrtImageFilter[image_type, image_type],
        input_offset=2.5,
    )
    check_unary_operator(
        itk.CudaAbsImageFilter[image_type], itk.AbsImageFilter[image_type, image_type]
    )
    check_unary_operator(
        itk.CudaAcosImageFilter[image_type],
        itk.AcosImageFilter[image_type, image_type],
        input_scale=0.1,
    )
    check_unary_operator(
        itk.CudaAsinImageFilter[image_type],
        itk.AsinImageFilter[image_type, image_type],
        input_scale=0.1,
    )
    check_unary_operator(
        itk.CudaAtanImageFilter[image_type], itk.AtanImageFilter[image_type, image_type]
    )
    check_unary_operator(
        itk.CudaBoundedReciprocalImageFilter[image_type],
        itk.BoundedReciprocalImageFilter[image_type, image_type],
    )
    check_unary_operator(
        itk.CudaCosImageFilter[image_type], itk.CosImageFilter[image_type, image_type]
    )
    check_unary_operator(
        itk.CudaExpImageFilter[image_type], itk.ExpImageFilter[image_type, image_type]
    )
    check_unary_operator(
        itk.CudaLogImageFilter[image_type],
        itk.LogImageFilter[image_type, image_type],
        input_offset=3.0,
    )
    check_unary_operator(
        itk.CudaLog10ImageFilter[image_type],
        itk.Log10ImageFilter[image_type, image_type],
        input_offset=3.0,
    )
    check_unary_operator(
        itk.CudaRoundImageFilter[image_type],
        itk.RoundImageFilter[image_type, image_type],
    )
    check_unary_operator(
        itk.CudaSinImageFilter[image_type], itk.SinImageFilter[image_type, image_type]
    )
    check_unary_operator(
        itk.CudaTanImageFilter[image_type], itk.TanImageFilter[image_type, image_type]
    )


def main():
    # Only floating-point pixel types are supported by the trigonometric and
    # rounding operators. Skip the pixel types that
    # CudaCommon_OPERATOR_INSTANTIATIONS_TYPES disabled so that the test does
    # not request wrappers that were not generated.
    tested = False
    for (image_type,) in itk.CudaSquareImageFilter.keys():
        pixel_type, dimension = itk.template(image_type)[1]
        if dimension != 2 or pixel_type not in (itk.F, itk.D):
            continue
        check_operators(image_type)
        tested = True
    if not tested:
        print("Skipping: no floating-point CudaImage types are wrapped")


if __name__ == "__main__":
    main()
