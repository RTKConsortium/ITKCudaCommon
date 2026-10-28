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
#include "itkCudaUtil.h"
#include "itkTestingMacros.h"
#include <itksys/SystemTools.hxx>
#include <iostream>

int
itkCudaDeviceSelectionTest(int, char *[])
{
  // Default behavior (no explicit set, no env)
  itksys::SystemTools::UnPutEnv("ITK_CUDA_DEFAULT_DEVICE");
  ITK_TEST_EXPECT_EQUAL(itk::GetDefaultCudaDevice(), itk::CudaGetMaxFlopsDev());

  // Env var fallback
  itksys::SystemTools::PutEnv("ITK_CUDA_DEFAULT_DEVICE=0");
  ITK_TEST_EXPECT_EQUAL(itk::GetDefaultCudaDevice(), 0);

  // Explicit set overrides env var
  itk::SetDefaultCudaDevice(0);
  ITK_TEST_EXPECT_EQUAL(itk::GetDefaultCudaDevice(), 0);

  auto mgr = itk::CudaDataManager::New();
  int  currentDevice = -1;
  itk::CudaCheckError(cudaGetDevice(&currentDevice));
  if (currentDevice != 0)
  {
    std::cerr << "Expected current device 0, got " << currentDevice << std::endl;
    return EXIT_FAILURE;
  }

  // Reset to auto
  itk::SetDefaultCudaDevice(-1);
  ITK_TEST_EXPECT_EQUAL(itk::GetDefaultCudaDevice(), itk::CudaGetMaxFlopsDev());

  auto mgrAuto = itk::CudaDataManager::New();
  itk::CudaCheckError(cudaGetDevice(&currentDevice));
  ITK_TEST_EXPECT_EQUAL(currentDevice, itk::CudaGetMaxFlopsDev());

  // Out-of-range device index
  ITK_TRY_EXPECT_EXCEPTION(itk::SetDefaultCudaDevice(1000000));

  std::cout << "CudaDeviceSelectionTest passed" << std::endl;
  return EXIT_SUCCESS;
}
