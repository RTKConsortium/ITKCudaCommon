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

#include "itkCudaImage.h"
#include "itkMath.h"

#include <iostream>
#include <vector>

using CudaImage2f = itk::CudaImage<float, 2>;

int
itkCudaImageRegionGraftTest(int, char *[])
{
  auto img = CudaImage2f::New();
  img->SetRegions(itk::MakeSize(32, 32));
  img->Allocate();

  // Initialize creates a new DataManager
  auto * dmBefore = img->GetCudaDataManager().GetPointer();
  img->Initialize();
  auto * dmAfter = img->GetCudaDataManager().GetPointer();
  if (dmBefore == dmAfter)
  {
    std::cerr << "Initialize should create a new DataManager" << std::endl;
    return EXIT_FAILURE;
  }

  // A grafted image shares the data manager of its source. When the data is
  // only on the GPU, reading the grafted image allocates the CPU buffer, which
  // must still work after the source is destroyed or released.
  for (const bool destroySource : { true, false })
  {
    auto source = CudaImage2f::New();
    source->SetRegions(itk::MakeSize(4, 4));
    const std::vector<float> values(16, 2.0f);
    itk::CUDA_CHECK(cudaMemcpy(source->GetCudaDataManager()->GetGPUBufferPointer(),
                               values.data(),
                               values.size() * sizeof(float),
                               cudaMemcpyHostToDevice));

    auto target = CudaImage2f::New();
    target->Graft(source);

    if (destroySource)
    {
      source = nullptr;
    }
    else
    {
      // As InPlaceImageFilter::ReleaseInputs() does
      source->ReleaseData();
    }
    const CudaImage2f * constTarget = target;
    const float *       buffer = constTarget->GetBufferPointer();
    if (buffer == nullptr)
    {
      std::cerr << "The CPU buffer of the grafted image should be allocated" << std::endl;
      return EXIT_FAILURE;
    }
    if (itk::Math::NotExactlyEquals(buffer[5], 2.0f) ||
        itk::Math::NotExactlyEquals(target->GetPixel(itk::MakeIndex(1, 1)), 2.0f))
    {
      std::cerr << "The grafted image should read the GPU data" << std::endl;
      return EXIT_FAILURE;
    }
  }

  std::cout << "CudaImageRegionGraftTest passed" << std::endl;
  return EXIT_SUCCESS;
}
