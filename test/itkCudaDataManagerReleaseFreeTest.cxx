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

#include "itkCudaDataManager.h"
#include <iostream>

int
itkCudaDataManagerReleaseFreeTest(int, char *[])
{
  constexpr size_t BUFFER_SIZE = 256 * sizeof(float);

  // SetGPUBufferDirty frees GPU memory when release is true (default)
  auto mgr = itk::CudaDataManager::New();
  mgr->SetBufferSize(BUFFER_SIZE);
  mgr->Allocate();
  mgr->SetGPUBufferDirty();
  if (mgr->GetGPUBufferSize() != 0)
  {
    std::cerr << "GPU buffer should be freed after SetGPUBufferDirty" << std::endl;
    return EXIT_FAILURE;
  }

  // SetGPUBufferDirty does not free when release is false
  auto mgr2 = itk::CudaDataManager::New();
  mgr2->SetReleaseDirtyGPUBuffer(false);
  mgr2->SetBufferSize(BUFFER_SIZE);
  mgr2->Allocate();
  size_t before = mgr2->GetGPUBufferSize();
  mgr2->SetGPUBufferDirty();
  if (mgr2->GetGPUBufferSize() != before)
  {
    std::cerr << "GPU buffer should not be freed when release is false" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "CudaDataManagerReleaseFreeTest passed" << std::endl;
  return EXIT_SUCCESS;
}
