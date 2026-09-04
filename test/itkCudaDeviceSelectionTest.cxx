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
#include <iostream>

int
itkCudaDeviceSelectionTest(int, char *[])
{
  // By default the device should be auto (-1).
  ITK_TEST_EXPECT_EQUAL(itk::GetDefaultCudaDevice(), -1);

  // Setting the default device to 0 and creating a manager should select device 0.
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

  // Resetting to auto (-1) makes the manager fall back to the max FLOPS device.
  itk::SetDefaultCudaDevice(-1);
  ITK_TEST_EXPECT_EQUAL(itk::GetDefaultCudaDevice(), -1);

  auto mgrAuto = itk::CudaDataManager::New();
  itk::CudaCheckError(cudaGetDevice(&currentDevice));
  ITK_TEST_EXPECT_EQUAL(currentDevice, itk::CudaGetMaxFlopsDev());

  // An out-of-range device index must throw an exception.
  ITK_TRY_EXPECT_EXCEPTION(itk::SetDefaultCudaDevice(1000000));

  std::cout << "CudaDeviceSelectionTest passed" << std::endl;
  return EXIT_SUCCESS;
}
