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

#include "itkCudaSquareImageFilter.h"
#include "itkCudaSqrtImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkMath.h"
#include "itkTestingMacros.h"

template <class TCudaFilter, class TCPUFilter>
int
CheckOperator(float inputOffset = 0.0f)
{
  using Image = typename TCudaFilter::InputImageType;
  auto input = Image::New();
  input->SetRegions(itk::MakeSize(17, 19));
  input->Allocate();

  const auto region = input->GetBufferedRegion();
  int        pixel = 0;
  for (itk::ImageRegionIterator<Image> it(input, region); !it.IsAtEnd(); ++it, ++pixel)
    it.Set(static_cast<typename Image::PixelType>(inputOffset + (pixel % 21 - 10) * 0.25f));

  auto reference = TCPUFilter::New();
  reference->SetInput(input);
  reference->InPlaceOff();
  reference->Update();

  auto filter = TCudaFilter::New();
  filter->SetInput(input);
  filter->InPlaceOff();
  filter->GPUEnabledOn();
  filter->Update();
  ITK_TEST_EXPECT_EQUAL(filter->GetOutput()->GetBufferedRegion(), region);

  itk::ImageRegionConstIterator<Image> actual(filter->GetOutput(), region);
  itk::ImageRegionConstIterator<Image> expected(reference->GetOutput(), region);
  for (; !actual.IsAtEnd(); ++actual, ++expected)
  {
    if (!itk::Math::AlmostEquals(actual.Get(), expected.Get()))
    {
      std::cerr << filter->GetNameOfClass() << " differs from ITK at " << actual.GetIndex() << std::endl;
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

int
itkCudaImageOperatorsTest(int, char *[])
{
  using Image = itk::CudaImage<float, 2>;
  // Add each CUDA operator and its ITK CPU counterpart here.
  ITK_TEST_EXPECT_EQUAL((CheckOperator<itk::CudaSquareImageFilter<Image>, itk::SquareImageFilter<Image, Image>>()),
                        EXIT_SUCCESS);
  ITK_TEST_EXPECT_EQUAL((CheckOperator<itk::CudaSqrtImageFilter<Image>, itk::SqrtImageFilter<Image, Image>>(2.5f)),
                        EXIT_SUCCESS);
  return EXIT_SUCCESS;
}
