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
#include <cmath>
#include <iostream>
#include <algorithm>
#include <array>

namespace
{
constexpr std::size_t NumberElts = 256;
constexpr std::size_t BufferSize = NumberElts * sizeof(float);
using BufferType = std::array<float, NumberElts>;
constexpr float FloatTolerance = 1e-6f;


/** Fill a device buffer with a single value.
 *
 * \param p_value Value assigned to every element of the device buffer.
 * \param p_gpuBufferPtr Device buffer to fill. Must not be nullptr.
 */
void
FillGPUBuffer(float p_value, void * p_gpuBufferPtr)
{
  BufferType gpuBuffHostCopy{};
  gpuBuffHostCopy.fill(p_value);

  itk::CUDA_CHECK(cudaMemcpy(p_gpuBufferPtr, gpuBuffHostCopy.data(), BufferSize, cudaMemcpyHostToDevice));
}


/** Check that all elements of an array are equal to the specified value
 *  within a floating-point tolerance.
 */
bool
IsSingleValue(float p_value, const BufferType & p_array)
{
  return std::all_of(p_array.begin(), p_array.end(), [&p_value](const float p_element) {
    return std::abs(p_value - p_element) <= FloatTolerance;
  });
}


/**
 * Check that all elements of a device buffer are equal to the specified value
 * within a floating-point tolerance.
 *
 * \param p_value Expected value for all buffer elements.
 * \param p_gpuBufferPtr Pointer to the device buffer to inspect. Must not be nullptr.
 */
bool
IsGPUBufferSingleValue(float p_value, void * p_gpuBufferPtr)
{
  // Copy GPU buffer to host
  BufferType gpuBuffHostCopy{};
  itk::CUDA_CHECK(cudaMemcpy(gpuBuffHostCopy.data(), p_gpuBufferPtr, BufferSize, cudaMemcpyDeviceToHost));

  return IsSingleValue(p_value, gpuBuffHostCopy);
}


} // namespace

int
itkCudaDataManagerDirtyFlagTest(int, char *[])
{

  BufferType cpuBuffer{};

  auto mgr = itk::CudaDataManager::New();
  mgr->SetBufferSize(BufferSize);
  mgr->Allocate();
  mgr->SetCPUBufferPointer(cpuBuffer.data());

  // Direct GPU buffer inspection is required to verify synchronization
  // behavior and buffer deallocation semantics.
  // Getting the buffer using mgr->GetGPUBufferPointer() triggers SetCPUBufferDirty()
  // which modifies the data manager internal state.
  void ** gpuBufferPointerPtr = static_cast<void **>(mgr->GetGPUBufferPointerPtr());

  // Both marked dirty:
  // Update() should throw and leave the dirty flags unchanged.
  mgr->SetCPUDirtyFlag(true);
  mgr->SetGPUDirtyFlag(true); // Frees GPU buffer
  ITK_TRY_EXPECT_EXCEPTION(mgr->Update());
  ITK_TEST_EXPECT_TRUE(*gpuBufferPointerPtr == nullptr);
  ITK_TEST_EXPECT_TRUE(mgr->IsCPUBufferDirty());
  ITK_TEST_EXPECT_TRUE(mgr->IsGPUBufferDirty());

  // Both buffers marked up-to-date:
  // Update() should do nothing.
  mgr->Allocate(); // Reallocate GPU buffer
  ITK_TEST_EXPECT_TRUE(*gpuBufferPointerPtr != nullptr);
  constexpr float CpuValueA = 1.0f;
  constexpr float GpuValueA = 2.0f;
  cpuBuffer.fill(CpuValueA);
  FillGPUBuffer(GpuValueA, *gpuBufferPointerPtr);
  mgr->SetGPUDirtyFlag(false);
  mgr->SetCPUDirtyFlag(false);
  mgr->Update();
  ITK_TEST_EXPECT_TRUE(IsSingleValue(CpuValueA, cpuBuffer));
  ITK_TEST_EXPECT_TRUE(IsGPUBufferSingleValue(GpuValueA, *gpuBufferPointerPtr));
  ITK_TEST_EXPECT_TRUE(!mgr->IsCPUBufferDirty());
  ITK_TEST_EXPECT_TRUE(!mgr->IsGPUBufferDirty());

  // Only GPU marked dirty:
  // Update() should regenerate GPU data from the CPU buffer.
  mgr->SetGPUDirtyFlag(true); // Frees GPU buffer
  ITK_TEST_EXPECT_TRUE(*gpuBufferPointerPtr == nullptr);
  mgr->SetCPUDirtyFlag(false);
  mgr->Update();
  ITK_TEST_EXPECT_TRUE(*gpuBufferPointerPtr != nullptr);
  ITK_TEST_EXPECT_TRUE(IsSingleValue(CpuValueA, cpuBuffer));
  ITK_TEST_EXPECT_TRUE(IsGPUBufferSingleValue(CpuValueA, *gpuBufferPointerPtr));
  ITK_TEST_EXPECT_TRUE(!mgr->IsCPUBufferDirty());
  ITK_TEST_EXPECT_TRUE(!mgr->IsGPUBufferDirty());

  // Only CPU marked dirty:
  // Update() should regenerate CPU data from the GPU buffer.
  constexpr float CpuValueB = 3.0f;
  constexpr float GpuValueB = 4.0f;
  cpuBuffer.fill(CpuValueB);
  FillGPUBuffer(GpuValueB, *gpuBufferPointerPtr);
  mgr->SetCPUDirtyFlag(true);
  mgr->SetGPUDirtyFlag(false);
  mgr->Update();
  ITK_TEST_EXPECT_TRUE(IsGPUBufferSingleValue(GpuValueB, *gpuBufferPointerPtr));
  ITK_TEST_EXPECT_TRUE(IsSingleValue(GpuValueB, cpuBuffer));
  ITK_TEST_EXPECT_TRUE(!mgr->IsCPUBufferDirty());
  ITK_TEST_EXPECT_TRUE(!mgr->IsGPUBufferDirty());

  // Only CPU marked dirty, but GPU buffer pointer is actually nullptr
  // Update() should not attempt to regenerate CPU data from the GPU buffer
  constexpr float CpuValueC = 5.0f;
  constexpr float GpuValueC = 6.0f;
  cpuBuffer.fill(CpuValueC);
  FillGPUBuffer(GpuValueC, *gpuBufferPointerPtr);
  mgr->SetGPUDirtyFlag(true); // Free GPU buffer
  mgr->SetGPUDirtyFlag(false);
  mgr->SetCPUDirtyFlag(true);
  mgr->Update();
  ITK_TEST_EXPECT_TRUE(*gpuBufferPointerPtr == nullptr);
  ITK_TEST_EXPECT_TRUE(IsSingleValue(CpuValueC, cpuBuffer));
  ITK_TEST_EXPECT_TRUE(mgr->IsCPUBufferDirty());
  ITK_TEST_EXPECT_TRUE(!mgr->IsGPUBufferDirty());

  std::cout << "CudaDataManagerDirtyFlagTest passed" << std::endl;
  return EXIT_SUCCESS;
}
