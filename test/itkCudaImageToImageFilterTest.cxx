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

#include "itkCudaImageToImageFilter.h"
#include "itkCudaImage.h"
#include "itkImage.h"
#include <cstdlib>
#include <cassert>
#include <typeinfo>

int
itkCudaImageToImageFilterTest(int, char *[])
{
  using CPUImageType = itk::Image<float, 2>;
  using GPUImageType = itk::CudaImage<float, 2>;

  auto filter = itk::CudaImageToImageFilter<GPUImageType, CPUImageType>::New();

  // Create a GPU image
  auto gpuImage = GPUImageType::New();
  gpuImage->SetRegions(itk::MakeSize(16, 16));
  gpuImage->Allocate();
  gpuImage->FillBuffer(1.0f);

  filter->SetInput(gpuImage);
  filter->Update();

  // Get the output and check its type
  auto        output = filter->GetOutput();
  std::string typeName = output->GetNameOfClass();
  assert(typeName == "Image");

  return EXIT_SUCCESS;
}
