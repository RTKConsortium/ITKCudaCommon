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
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkMath.h"
#include "itkTestingMacros.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <type_traits>

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

template <class TPixel>
int
CheckOperators()
{
  using Image = itk::CudaImage<TPixel, 2>;
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
  return EXIT_SUCCESS;
}

int
itkCudaImageOperatorsTest(int, char *[])
{
  // The unary operators are only instantiated for the configured floating-point
  // pixels. Skip the pixel types that CudaCommon_OPERATOR_INSTANTIATIONS_TYPES
  // disabled so that the test links against the available operators.
#ifdef CudaCommon_TEST_float
  ITK_TEST_EXPECT_EQUAL(CheckOperators<float>(), EXIT_SUCCESS);
#endif
#ifdef CudaCommon_TEST_double
  ITK_TEST_EXPECT_EQUAL(CheckOperators<double>(), EXIT_SUCCESS);
#endif
  return EXIT_SUCCESS;
}
