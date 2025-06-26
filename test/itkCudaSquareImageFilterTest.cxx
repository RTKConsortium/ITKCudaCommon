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

/**
 * Test program for itkCudaImage class.
 * This program shows how to use Cuda image and Cuda filter.
 */
#include "itkCudaImage.h"
#include "itkCudaSquareImageFilter.h"

using ItkImage2f = itk::CudaImage<float, 2>;

int
itkCudaSquareImageFilterTest(int, char *[])
{
  // create
  float fillValue = 2.0f;
  auto src = ItkImage2f::New();
  src->SetRegions(itk::MakeSize(256, 256));
  src->Allocate();
  src->FillBuffer(fillValue);

  auto sqImageFilter = itk::CudaSquareImageFilter<ItkImage2f>::New();
  sqImageFilter->SetInput(src);
  sqImageFilter->Update();

  // check pixel value
  if (sqImageFilter->GetOutput()->GetPixel(itk::MakeIndex(0, 0)) != fillValue * fillValue)
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
