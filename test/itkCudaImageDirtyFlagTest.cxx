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
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkTestingMacros.h"

// Check that read-only CPU access to a CudaImage keeps its GPU buffer valid
// while writable CPU access invalidates it, see
// https://github.com/RTKConsortium/ITKCudaCommon/issues/107
int
itkCudaImageDirtyFlagTest(int, char *[])
{
  using ImageType = itk::CudaImage<float, 2>;

  const auto region = itk::ImageRegion<2>(itk::MakeSize(4, 4));

  auto image = ImageType::New();
  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(1.0f);
  image->UpdateBuffers();

  const auto mgr = image->GetCudaDataManager();
  ITK_TEST_EXPECT_TRUE(!mgr->IsCPUBufferDirty());
  ITK_TEST_EXPECT_TRUE(!mgr->IsGPUBufferDirty());

  // Read-only access must not invalidate the GPU buffer
  const ImageType * constImage = image;
  for (itk::ImageRegionConstIterator<ImageType> it(constImage, region); !it.IsAtEnd(); ++it)
  {
    ITK_TEST_EXPECT_EQUAL(it.Get(), 1.0f);
  }
  ITK_TEST_EXPECT_TRUE(!mgr->IsGPUBufferDirty());
  ITK_TEST_EXPECT_TRUE(!mgr->IsCPUBufferDirty());

  // Writable access must invalidate the GPU buffer
  for (itk::ImageRegionIterator<ImageType> it(image, region); !it.IsAtEnd(); ++it)
  {
    it.Set(2.0f);
  }
  ITK_TEST_EXPECT_TRUE(mgr->IsGPUBufferDirty());
  ITK_TEST_EXPECT_TRUE(!mgr->IsCPUBufferDirty());

  // The GPU buffer is regenerated from the CPU buffer
  image->UpdateBuffers();
  ITK_TEST_EXPECT_TRUE(!mgr->IsGPUBufferDirty());

  std::cout << "Test finished." << std::endl;
  return EXIT_SUCCESS;
}
