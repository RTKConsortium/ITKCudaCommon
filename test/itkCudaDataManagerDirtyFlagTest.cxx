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
#include <cmath>
#include <iostream>

int
itkCudaDataManagerDirtyFlagTest(int, char *[])
{
  constexpr size_t BUFFER_SIZE = 256 * sizeof(float);

  auto mgr = itk::CudaDataManager::New();
  mgr->SetBufferSize(BUFFER_SIZE);
  mgr->Allocate();
  auto * cpuBuf = new float[256];
  mgr->SetCPUBufferPointer(cpuBuf);

  // Both-dirty error in Update
  mgr->SetCPUDirtyFlag(true);
  mgr->SetGPUDirtyFlag(true);
  bool threw = false;
  try
  {
    mgr->Update();
  }
  catch (itk::ExceptionObject &)
  {
    threw = true;
  }
  if (!threw)
  {
    std::cerr << "Update should throw when both buffers are dirty" << std::endl;
    delete[] cpuBuf;
    return EXIT_FAILURE;
  }

  // Update with GPU dirty only: re-allocates GPU buffer and copies CPU→GPU
  mgr->SetGPUDirtyFlag(false);
  mgr->SetCPUDirtyFlag(false);
  for (int i = 0; i < 256; ++i)
    cpuBuf[i] = static_cast<float>(i);
  mgr->Update(); // sync both — GPU now holds {0,1,2,...}
  mgr->SetGPUDirtyFlag(true); // frees GPU buffer
  mgr->SetCPUDirtyFlag(false);
  mgr->Update();
  if (std::abs(cpuBuf[0] - 0.0f) >= 1e-6f)
  {
    std::cerr << "CPU buffer should be unchanged after CPU→GPU copy" << std::endl;
    delete[] cpuBuf;
    return EXIT_FAILURE;
  }

  // Update with CPU dirty only: copies GPU→CPU
  mgr->SetCPUDirtyFlag(false);
  mgr->SetGPUDirtyFlag(false);
  for (int i = 0; i < 256; ++i)
    cpuBuf[i] = static_cast<float>(i);
  mgr->Update(); // sync both — GPU now holds {0,1,2,...}
  for (int i = 0; i < 256; ++i)
    cpuBuf[i] = 99.0f; // make CPU stale
  mgr->SetCPUDirtyFlag(true);
  mgr->SetGPUDirtyFlag(false);
  mgr->Update();
  if (std::abs(cpuBuf[0] - 0.0f) >= 1e-6f)
  {
    std::cerr << "CPU buffer should have GPU values after GPU→CPU copy" << std::endl;
    delete[] cpuBuf;
    return EXIT_FAILURE;
  }
  delete[] cpuBuf;

  std::cout << "CudaDataManagerDirtyFlagTest passed" << std::endl;
  return EXIT_SUCCESS;
}
