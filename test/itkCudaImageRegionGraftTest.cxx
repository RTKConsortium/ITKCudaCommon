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

  // A grafted image shares the data manager of its source, whose image pointer
  // must not dangle when the source is released before the grafted image.
  using DataManagerType = itk::CudaImageDataManager<CudaImage2f>;
  const auto getImagePointer = [](const CudaImage2f * image) {
    return static_cast<DataManagerType *>(image->GetCudaDataManager().GetPointer())->GetImagePointer();
  };
  for (const bool destroySource : { true, false })
  {
    auto source = CudaImage2f::New();
    source->SetRegions(itk::MakeSize(4, 4));
    source->Allocate();
    source->FillBuffer(2.0f);
    source->GetCudaDataManager()->GetGPUBufferPointer();

    auto target = CudaImage2f::New();
    target->Graft(source);
    if (getImagePointer(target) != source.GetPointer())
    {
      std::cerr << "The source should remain the image pointer after a graft" << std::endl;
      return EXIT_FAILURE;
    }

    if (destroySource)
    {
      source = nullptr;
    }
    else
    {
      // As InPlaceImageFilter::ReleaseInputs() does
      source->ReleaseData();
    }
    if (getImagePointer(target) != target.GetPointer())
    {
      std::cerr << "The grafted image should become the image pointer" << std::endl;
      return EXIT_FAILURE;
    }
    if (itk::Math::NotExactlyEquals(target->GetPixel(itk::MakeIndex(1, 1)), 2.0f))
    {
      std::cerr << "The grafted image should keep the data" << std::endl;
      return EXIT_FAILURE;
    }
  }

  std::cout << "CudaImageRegionGraftTest passed" << std::endl;
  return EXIT_SUCCESS;
}
