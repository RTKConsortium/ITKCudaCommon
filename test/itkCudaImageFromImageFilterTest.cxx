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

#include "itkCudaImageFromImageFilter.h"
#include "itkCudaImage.h"
#include "itkImageRegionIterator.h"
#include <iostream>
#include <cstring>

int
itkCudaImageFromImageFilterTest(int, char *[])
{
  using InputImageType = itk::Image<float, 3>;
  auto                       input = InputImageType::New();
  InputImageType::RegionType region;
  InputImageType::IndexType  start{ { 0, 0, 0 } };
  InputImageType::SizeType   size{ { 4, 4, 4 } };
  region.SetIndex(start);
  region.SetSize(size);
  input->SetRegions(region);
  input->Allocate();
  input->FillBuffer(1.0f);
  auto filter = itk::CudaImageFromImageFilter<InputImageType>::New();
  filter->SetInput(input);

  // Prepare a smaller output requested region (simulate downstream streaming request)
  InputImageType::RegionType smallRegion = region;
  InputImageType::SizeType   smallSize = size;
  smallSize[0] /= 2;
  smallRegion.SetSize(smallSize);

  auto * out = filter->GetOutput();
  out->SetRequestedRegion(smallRegion);

  filter->Update();

  if (input->GetRequestedRegion() != smallRegion)
  {
    std::cerr << "Input requested region did not match output requested region propagation\n";
    return EXIT_FAILURE;
  }

  // Check output dynamic type is CudaImage
  if (std::string(out->GetNameOfClass()) != "CudaImage")
  {
    std::cerr << "Output is not an itk::CudaImage (got " << out->GetNameOfClass() << ")" << std::endl;
    return EXIT_FAILURE;
  }

  // Check pixel container pointer equality
  if (out->GetPixelContainer() != input->GetPixelContainer())
  {
    std::cerr << "PixelContainer was not shared" << std::endl;
    return EXIT_FAILURE;
  }

  // Mutate one pixel through output
  auto * container = out->GetPixelContainer();
  auto * buffer = container->GetBufferPointer();
  buffer[0] = 5.0f;
  if (input->GetBufferPointer()[0] != 5.0f)
  {
    std::cerr << "Modification did not reflect in input" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Test passed" << std::endl;
  return EXIT_SUCCESS;
}
