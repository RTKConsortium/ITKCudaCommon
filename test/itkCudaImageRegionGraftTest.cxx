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

  std::cout << "CudaImageRegionGraftTest passed" << std::endl;
  return EXIT_SUCCESS;
}
