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

#include "itkCudaMemoryProbe.h"
#include "itkCudaImage.h"
#include <iostream>

int
itkCudaMemoryProbeTest(int, char *[])
{
  itk::CudaMemoryProbe probe;
  auto                 img = itk::CudaImage<float, 3>::New();
  img->SetRegions(itk::MakeSize(128, 128, 128));

  // allocate CPU and prepare CPU data
  img->Allocate();
  img->FillBuffer(1.0f);

  probe.Start();
  img->UpdateBuffers();
  probe.Stop();

  // Compute the expected GPU allocation size directly from image size.
  // numPixels * sizeof(float) / 1024 -> kB
  size_t gpuBytes = img->GetCudaDataManager()->GetGPUBufferSize();
  double expectedKB = static_cast<double>(gpuBytes / 1024);

  double    mean = probe.GetMean();
  long long observedKB = static_cast<long long>(expectedKB);

  if (mean != expectedKB)
  {
    std::cerr << "Probe mean mismatch: " << mean << " (expected: " << expectedKB << ")" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
