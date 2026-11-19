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
#include "itkTestingMacros.h"

#include <cstdint>
#include <iostream>

namespace
{
class MutexDataManager : public itk::CudaDataManager
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MutexDataManager);
  using Self = MutexDataManager;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro(Self);

  /** Tell whether the mutex is locked and unlock it if it is, so that the
   * test can report the failure instead of hanging. */
  bool
  UnlockMutexIfLocked()
  {
    if (m_Mutex.try_lock())
    {
      m_Mutex.unlock();
      return false;
    }
    m_Mutex.unlock();
    return true;
  }

protected:
  MutexDataManager() = default;
};
} // namespace

int
itkCudaDataManagerMutexTest(int, char *[])
{
  // A failed GPU allocation must not leave the mutex locked. Otherwise, the
  // next lock, e.g., in the destructor of a CudaImage, never returns.
  auto mgr = MutexDataManager::New();
  mgr->SetBufferSize(std::size_t{ 1 } << 60);
  mgr->SetGPUDirtyFlag(true);
  ITK_TRY_EXPECT_EXCEPTION(mgr->UpdateGPUBuffer());
  ITK_TEST_EXPECT_TRUE(!mgr->UnlockMutexIfLocked());

  // The data manager is still usable after the failure.
  mgr->SetBufferSize(256 * sizeof(float));
  mgr->SetGPUDirtyFlag(true);
  ITK_TRY_EXPECT_NO_EXCEPTION(mgr->UpdateGPUBuffer());
  ITK_TEST_EXPECT_TRUE(mgr->GetGPUBufferSize() == 256 * sizeof(float));

  std::cout << "Test finished." << std::endl;
  return EXIT_SUCCESS;
}
