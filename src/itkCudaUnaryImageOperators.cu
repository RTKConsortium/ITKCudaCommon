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
// Omit ITK DLL annotations when compiling CUDA, as in RTK's CUDA utilities.
#ifndef ITK_STATIC
#  define ITK_STATIC
#  include "itkMacro.h"
#  undef ITK_STATIC
#endif

#include "itkCudaUnaryImageOperators.hcu"
#include "itkCudaImageOperators.hcu"

#include <cmath>

namespace itk
{

template <class PixelType>
void
CudaSquareImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto square = [] __device__(PixelType value) { return static_cast<PixelType>(value * value); };
  LaunchCudaImageOperation(numberOfPixels, square, output, input);
}

ITK_CUDA_IMAGE_OPERATOR_INSTANTIATE(CudaSquareImage)

template <class PixelType>
void
CudaSqrtImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto sqrtOperation = [] __device__(PixelType value) {
    return static_cast<PixelType>(sqrt(static_cast<double>(value)));
  };
  LaunchCudaImageOperation(numberOfPixels, sqrtOperation, output, input);
}

ITK_CUDA_IMAGE_OPERATOR_INSTANTIATE(CudaSqrtImage)

} // namespace itk
