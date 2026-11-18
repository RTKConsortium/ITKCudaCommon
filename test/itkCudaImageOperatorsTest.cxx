/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "itkCudaAbsImageFilter.h"
#include "itkCudaAcosImageFilter.h"
#include "itkCudaAsinImageFilter.h"
#include "itkCudaAtanImageFilter.h"
#include "itkCudaBoundedReciprocalImageFilter.h"
#include "itkCudaCosImageFilter.h"
#include "itkCudaExpImageFilter.h"
#include "itkCudaLogImageFilter.h"
#include "itkCudaLog10ImageFilter.h"
#include "itkCudaRoundImageFilter.h"
#include "itkCudaSinImageFilter.h"
#include "itkCudaTanImageFilter.h"
#include "itkCudaSquareImageFilter.h"
#include "itkCudaSqrtImageFilter.h"
#include "itkCudaAddImageFilter.h"
#include "itkCudaSubtractImageFilter.h"
#include "itkCudaMultiplyImageFilter.h"
#include "itkCudaDivideImageFilter.h"
#include "itkCudaMaximumImageFilter.h"
#include "itkCudaMinimumImageFilter.h"
#include "itkCudaPowImageFilter.h"
#include "itkCudaAtan2ImageFilter.h"
#include "itkCudaDivideOrZeroOutImageFilter.h"
#include "itkCudaWeightedAddImageFilter.h"
#include "itkCudaModulusImageFilter.h"
#include "itkCudaAndImageFilter.h"
#include "itkCudaOrImageFilter.h"
#include "itkCudaXorImageFilter.h"
#include "itkCudaBinaryMagnitudeImageFilter.h"
#include "itkCudaConstrainedValueAdditionImageFilter.h"
#include "itkCudaConstrainedValueDifferenceImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkMath.h"
#include "itkTestingMacros.h"

#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <type_traits>
#include <utility>

template <class TImage>
int
CheckImageQuality(const TImage * image, const TImage * reference)
{
  ITK_TEST_EXPECT_EQUAL(image->GetBufferedRegion(), reference->GetBufferedRegion());

  itk::ImageRegionConstIterator<TImage> actual(image, image->GetBufferedRegion());
  itk::ImageRegionConstIterator<TImage> expected(reference, reference->GetBufferedRegion());
  for (; !actual.IsAtEnd(); ++actual, ++expected)
  {
    const auto actualValue = actual.Get();
    const auto expectedValue = expected.Get();
    ITK_TEST_EXPECT_TRUE(actualValue == expectedValue || itk::Math::AlmostEquals(actualValue, expectedValue));
    if (actualValue == 0 && expectedValue == 0)
    {
      ITK_TEST_EXPECT_EQUAL(std::signbit(actualValue), std::signbit(expectedValue));
    }
  }
  return EXIT_SUCCESS;
}

template <class TCudaFilter, class TCPUFilter>
int
CheckUnaryOperator(float inputOffset = 0.0f, float inputScale = 0.25f)
{
  using Image = typename TCudaFilter::InputImageType;
  auto input = Image::New();
  input->SetRegions(itk::MakeSize(17, 19));
  input->Allocate();

  const auto region = input->GetBufferedRegion();
  int        pixel = 0;
  for (itk::ImageRegionIterator<Image> it(input, region); !it.IsAtEnd(); ++it, ++pixel)
  {
    it.Set(static_cast<typename Image::PixelType>(inputOffset + (pixel % 21 - 10) * inputScale));
  }
  if constexpr (std::is_same_v<TCudaFilter, itk::CudaAbsImageFilter<Image>> ||
                std::is_same_v<TCudaFilter, itk::CudaRoundImageFilter<Image>>)
  {
    input->SetPixel(itk::MakeIndex(0, 0), -0.0f);
    input->SetPixel(itk::MakeIndex(1, 0), std::nextafter(0.5f, 0.0f));
  }

  auto reference = TCPUFilter::New();
  reference->SetInput(input);
  reference->InPlaceOff();
  ITK_TRY_EXPECT_NO_EXCEPTION(reference->Update());

  auto filter = TCudaFilter::New();
  filter->SetInput(input);
  filter->InPlaceOff();
  filter->GPUEnabledOn();
  ITK_TRY_EXPECT_NO_EXCEPTION(filter->Update());
  ITK_TEST_EXPECT_EQUAL(filter->GetOutput()->GetBufferedRegion(), region);

  std::cout << "Checking " << filter->GetNameOfClass() << " against " << reference->GetNameOfClass() << std::endl;
  return CheckImageQuality<Image>(filter->GetOutput(), reference->GetOutput());
}

template <class TCudaFilter, class TCPUFilter>
int
CheckConstrainedValueBoundaries(
  std::initializer_list<
    std::pair<typename TCudaFilter::InputImageType::PixelType, typename TCudaFilter::InputImageType::PixelType>> values)
{
  using Image = typename TCudaFilter::InputImageType;
  auto input = Image::New();
  auto second = Image::New();
  input->SetRegions(itk::MakeSize(static_cast<itk::SizeValueType>(values.size()), 1));
  second->SetRegions(input->GetLargestPossibleRegion());
  input->Allocate();
  second->Allocate();

  itk::SizeValueType pixel = 0;
  for (const auto & [first, secondValue] : values)
  {
    input->GetBufferPointer()[pixel] = first;
    second->GetBufferPointer()[pixel] = secondValue;
    ++pixel;
  }

  auto reference = TCPUFilter::New();
  reference->SetInput1(input);
  reference->SetInput2(second);
  reference->InPlaceOff();
  ITK_TRY_EXPECT_NO_EXCEPTION(reference->Update());

  auto filter = TCudaFilter::New();
  filter->SetInput1(input);
  filter->SetInput2(second);
  filter->InPlaceOff();
  filter->GPUEnabledOn();
  ITK_TRY_EXPECT_NO_EXCEPTION(filter->Update());

  return CheckImageQuality<Image>(filter->GetOutput(), reference->GetOutput());
}

template <class TCudaFilter, class TCPUFilter, class TConfigure>
int
CheckBinaryOperator(unsigned int constantInput, TConfigure configure)
{
  using Image = typename TCudaFilter::InputImageType;
  using SecondImage = typename TCudaFilter::Input2ImageType;
  auto input = Image::New();
  auto second = SecondImage::New();
  input->SetRegions(itk::MakeSize(17, 19));
  second->SetRegions(input->GetLargestPossibleRegion());
  input->Allocate();
  second->Allocate();
  for (unsigned int i = 0; i < 17 * 19; ++i)
  {
    // Includes 3^1 for the integer-power regression and crosses a CUDA block.
    input->GetBufferPointer()[i] = static_cast<typename Image::PixelType>(i % 5 + 1);
    second->GetBufferPointer()[i] = static_cast<typename SecondImage::PixelType>(i % 4);
  }

  const auto setInputs = [&](auto filter) {
    if (constantInput == 1)
    {
      filter->SetConstant1(2.0f);
    }
    else
    {
      filter->SetInput1(input);
    }
    if (constantInput == 2)
    {
      filter->SetConstant2(2.0f);
    }
    else
    {
      filter->SetInput2(second);
    }
    filter->InPlaceOff();
    configure(filter);
  };

  auto reference = TCPUFilter::New();
  setInputs(reference);
  ITK_TRY_EXPECT_NO_EXCEPTION(reference->Update());

  auto filter = TCudaFilter::New();
  setInputs(filter);
  filter->GPUEnabledOn();
  ITK_TRY_EXPECT_NO_EXCEPTION(filter->Update());
  std::cout << "Checking " << filter->GetNameOfClass() << " against " << reference->GetNameOfClass() << std::endl;
  return CheckImageQuality<Image>(filter->GetOutput(), reference->GetOutput());
}

int
itkCudaImageOperatorsTest(int, char *[])
{
  using Image = itk::CudaImage<float, 2>;
  using IntegerImage = itk::CudaImage<short, 2>;
  using UnsignedImage = itk::CudaImage<unsigned char, 2>;
  ITK_TEST_EXPECT_EQUAL((CheckUnaryOperator<itk::CudaSquareImageFilter<Image>, itk::SquareImageFilter<Image, Image>>()),
                        EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL((CheckUnaryOperator<itk::CudaSqrtImageFilter<Image>, itk::SqrtImageFilter<Image, Image>>(2.5f)),
                        EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL((CheckUnaryOperator<itk::CudaAbsImageFilter<Image>, itk::AbsImageFilter<Image, Image>>()),
                        EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL(
    (CheckUnaryOperator<itk::CudaAcosImageFilter<Image>, itk::AcosImageFilter<Image, Image>>(0.0f, 0.1f)),
    EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL(
    (CheckUnaryOperator<itk::CudaAsinImageFilter<Image>, itk::AsinImageFilter<Image, Image>>(0.0f, 0.1f)),
    EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL((CheckUnaryOperator<itk::CudaAtanImageFilter<Image>, itk::AtanImageFilter<Image, Image>>()),
                        EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL((CheckUnaryOperator<itk::CudaBoundedReciprocalImageFilter<Image>,
                                            itk::BoundedReciprocalImageFilter<Image, Image>>()),
                        EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL((CheckUnaryOperator<itk::CudaCosImageFilter<Image>, itk::CosImageFilter<Image, Image>>()),
                        EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL((CheckUnaryOperator<itk::CudaExpImageFilter<Image>, itk::ExpImageFilter<Image, Image>>()),
                        EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL((CheckUnaryOperator<itk::CudaLogImageFilter<Image>, itk::LogImageFilter<Image, Image>>(3.0f)),
                        EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL(
    (CheckUnaryOperator<itk::CudaLog10ImageFilter<Image>, itk::Log10ImageFilter<Image, Image>>(3.0f)), EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL((CheckUnaryOperator<itk::CudaRoundImageFilter<Image>, itk::RoundImageFilter<Image, Image>>()),
                        EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL((CheckUnaryOperator<itk::CudaSinImageFilter<Image>, itk::SinImageFilter<Image, Image>>()),
                        EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL((CheckUnaryOperator<itk::CudaTanImageFilter<Image>, itk::TanImageFilter<Image, Image>>()),
                        EXIT_SUCCESS);
  // The loop below only uses small values, so exercise the limits of the
  // output type here.
  ITK_TEST_EXPECT_EQUAL(
    (CheckConstrainedValueBoundaries<
      itk::CudaConstrainedValueAdditionImageFilter<IntegerImage>,
      itk::ConstrainedValueAdditionImageFilter<IntegerImage, IntegerImage, IntegerImage>>(
      { { 30000, 30000 }, { -30000, -30000 }, { 30000, -30000 }, { -30000, 30000 }, { 32767, 1 }, { -32768, 1 } })),
    EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL(
    (CheckConstrainedValueBoundaries<
      itk::CudaConstrainedValueDifferenceImageFilter<IntegerImage>,
      itk::ConstrainedValueDifferenceImageFilter<IntegerImage, IntegerImage, IntegerImage>>(
      { { 30000, 30000 }, { -30000, -30000 }, { 30000, -30000 }, { -30000, 30000 }, { 32767, 1 }, { -32768, 1 } })),
    EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL((CheckConstrainedValueBoundaries<
                          itk::CudaConstrainedValueAdditionImageFilter<UnsignedImage>,
                          itk::ConstrainedValueAdditionImageFilter<UnsignedImage, UnsignedImage, UnsignedImage>>(
                          { { 200, 200 }, { 0, 200 }, { 200, 0 }, { 255, 1 }, { 1, 255 } })),
                        EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL((CheckConstrainedValueBoundaries<
                          itk::CudaConstrainedValueDifferenceImageFilter<UnsignedImage>,
                          itk::ConstrainedValueDifferenceImageFilter<UnsignedImage, UnsignedImage, UnsignedImage>>(
                          { { 200, 200 }, { 0, 200 }, { 200, 0 }, { 255, 1 }, { 1, 255 } })),
                        EXIT_SUCCESS);

  // The default threshold for integer pixels is 1, so a zero denominator
  // gives the constant. The CPU filter is not the reference because ITK 5
  // truncates its default threshold, 1e-5, to 0 and divides by zero.
  {
    auto numerator = IntegerImage::New();
    numerator->SetRegions(itk::MakeSize(4, 1));
    numerator->Allocate();
    auto denominator = IntegerImage::New();
    denominator->SetRegions(numerator->GetBufferedRegion());
    denominator->Allocate();
    const short numerators[] = { 7, -7, 8, 9 };
    const short denominators[] = { 0, 0, 2, -3 };
    const short expected[] = { 5, 5, 4, 5 };
    for (unsigned int i = 0; i < 4; ++i)
    {
      numerator->GetBufferPointer()[i] = numerators[i];
      denominator->GetBufferPointer()[i] = denominators[i];
    }

    auto filter = itk::CudaDivideOrZeroOutImageFilter<IntegerImage>::New();
    ITK_TEST_EXPECT_EQUAL(filter->GetThreshold(), 1);
    filter->SetInput1(numerator);
    filter->SetInput2(denominator);
    filter->SetConstant(5);
    filter->InPlaceOff();
    ITK_TRY_EXPECT_NO_EXCEPTION(filter->Update());
    for (unsigned int i = 0; i < 4; ++i)
    {
      ITK_TEST_EXPECT_EQUAL(filter->GetOutput()->GetBufferPointer()[i], expected[i]);
    }
  }

  const auto defaults = [](auto) {};
  // Set alpha through the ITK base class, whose SetAlpha() is not virtual.
  const auto weighted = [](auto filter) {
    itk::WeightedAddImageFilter<Image, Image, Image> * base = filter.GetPointer();
    base->SetAlpha(0.25);
  };
  const auto division = [](auto filter) {
    filter->SetThreshold(1);
    filter->SetConstant(7);
  };

  // Image/image, constant/image, and image/constant use the same small dataset.
  for (unsigned int constantInput : { 0, 1, 2 })
  {
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaAddImageFilter<Image>, itk::AddImageFilter<Image, Image, Image>>(constantInput,
                                                                                                     defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaSubtractImageFilter<Image>, itk::SubtractImageFilter<Image, Image, Image>>(
        constantInput, defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaMultiplyImageFilter<Image>, itk::MultiplyImageFilter<Image, Image, Image>>(
        constantInput, defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaDivideImageFilter<Image>, itk::DivideImageFilter<Image, Image, Image>>(
        constantInput, defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaMaximumImageFilter<Image>, itk::MaximumImageFilter<Image, Image, Image>>(
        constantInput, defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaMinimumImageFilter<Image>, itk::MinimumImageFilter<Image, Image, Image>>(
        constantInput, defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaPowImageFilter<Image>, itk::PowImageFilter<Image, Image, Image>>(constantInput,
                                                                                                     defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaAtan2ImageFilter<Image>, itk::Atan2ImageFilter<Image, Image, Image>>(constantInput,
                                                                                                         defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaBinaryMagnitudeImageFilter<Image>,
                           itk::BinaryMagnitudeImageFilter<Image, Image, Image>>(constantInput, defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaConstrainedValueAdditionImageFilter<Image>,
                           itk::ConstrainedValueAdditionImageFilter<Image, Image, Image>>(constantInput, defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaConstrainedValueDifferenceImageFilter<Image>,
                           itk::ConstrainedValueDifferenceImageFilter<Image, Image, Image>>(constantInput, defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaWeightedAddImageFilter<Image>, itk::WeightedAddImageFilter<Image, Image, Image>>(
        constantInput, weighted)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaDivideOrZeroOutImageFilter<Image>,
                           itk::DivideOrZeroOutImageFilter<Image, Image, Image>>(constantInput, division)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaModulusImageFilter<IntegerImage>,
                           itk::ModulusImageFilter<IntegerImage, IntegerImage, IntegerImage>>(constantInput, defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaAndImageFilter<IntegerImage>,
                           itk::AndImageFilter<IntegerImage, IntegerImage, IntegerImage>>(constantInput, defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaOrImageFilter<IntegerImage>,
                           itk::OrImageFilter<IntegerImage, IntegerImage, IntegerImage>>(constantInput, defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaXorImageFilter<IntegerImage>,
                           itk::XorImageFilter<IntegerImage, IntegerImage, IntegerImage>>(constantInput, defaults)),
      EXIT_SUCCESS);
    ITK_TEST_EXPECT_EQUAL(
      (CheckBinaryOperator<itk::CudaPowImageFilter<IntegerImage>,
                           itk::PowImageFilter<IntegerImage, IntegerImage, IntegerImage>>(constantInput, defaults)),
      EXIT_SUCCESS);
  }
  return EXIT_SUCCESS;
}
