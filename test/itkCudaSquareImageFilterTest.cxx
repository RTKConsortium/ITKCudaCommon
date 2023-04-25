/*=========================================================================
*
*  Copyright Insight Software Consortium
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

int itkCudaSquareImageFilterTest(int, char*[])
{
  unsigned int width, height;
  width = 256;
  height = 256;

  // set size & region
  ItkImage2f::Pointer src;

  ItkImage2f::IndexType start;
  start[0] = 0;
  start[1] = 0;
  ItkImage2f::SizeType size;
  size[0] = width;
  size[1] = height;
  ItkImage2f::RegionType region;
  region.SetSize(size);
  region.SetIndex(start);

  // create
  float fillValue = 2.0f;
  src = ItkImage2f::New();
  src->SetRegions(region);
  src->Allocate();
  src->FillBuffer(fillValue);

  itk::CudaSquareImageFilter<ItkImage2f>::Pointer sqImageFilter =
    itk::CudaSquareImageFilter<ItkImage2f>::New();
  sqImageFilter->SetInput(src);
  sqImageFilter->Update();

  // check pixel value
  ItkImage2f::IndexType idx;
  idx[0] = 0;
  idx[1] = 0;

  if (sqImageFilter->GetOutput()->GetPixel(idx) != fillValue * fillValue)
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
