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
// Omit ITK DLL annotations when compiling CUDA.
#ifndef ITK_STATIC
#  define ITK_STATIC
#  include "itkMacro.h"
#  undef ITK_STATIC
#endif

#include "itkCudaBinaryGeneratorImageFilter.hcu"
#include "itkCudaUtil.h"

#include "CudaCommonExport.h"

#include <cmath>
#include <type_traits>

namespace itk
{
namespace
{

template <typename TPixel>
__device__ bool
CudaValueIsAlmostZero(TPixel value, TPixel tolerance)
{
  if constexpr (std::is_floating_point_v<TPixel>)
  {
    return ::fabs(value) <= tolerance;
  }
  else
  {
    return value == TPixel{};
  }
}

template <CudaBinaryImageOperation VOperation, typename TInputPixel1, typename TInputPixel2, typename TOutputPixel>
__global__ void
CudaBinaryImageFilterKernel(itk::SizeValueType   numberOfPixels,
                            const TInputPixel1 * input1,
                            const TInputPixel2 * input2,
                            TInputPixel1         constant1,
                            TInputPixel2         constant2,
                            TOutputPixel *       output,
                            TInputPixel2         divideAlmostZeroTolerance,
                            TInputPixel2         divideOrZeroOutThreshold,
                            TOutputPixel         divideOrZeroOutConstant,
                            TOutputPixel         divideByZeroValue)
{
  const itk::SizeValueType index = static_cast<itk::SizeValueType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= numberOfPixels)
  {
    return;
  }
  // A null input buffer means that the input is the constant.
  TInputPixel1 value1 = constant1;
  TInputPixel2 value2 = constant2;
  if (input1 != nullptr)
  {
    value1 = input1[index];
  }
  if (input2 != nullptr)
  {
    value2 = input2[index];
  }

  if constexpr (VOperation == CudaBinaryImageOperation::Add)
  {
    output[index] = static_cast<TOutputPixel>(value1 + value2);
  }
  else if constexpr (VOperation == CudaBinaryImageOperation::Subtract)
  {
    output[index] = static_cast<TOutputPixel>(value1 - value2);
  }
  else if constexpr (VOperation == CudaBinaryImageOperation::Multiply)
  {
    output[index] = static_cast<TOutputPixel>(value1 * value2);
  }
  else if constexpr (VOperation == CudaBinaryImageOperation::Divide)
  {
    output[index] = CudaValueIsAlmostZero(value2, divideAlmostZeroTolerance)
                      ? divideByZeroValue
                      : static_cast<TOutputPixel>(value1 / value2);
  }
  else if constexpr (VOperation == CudaBinaryImageOperation::DivideOrZeroOut)
  {
    output[index] = value2 < divideOrZeroOutThreshold
                      ? divideOrZeroOutConstant
                      : static_cast<TOutputPixel>(value1) / static_cast<TOutputPixel>(value2);
  }
}

} // namespace

template <CudaBinaryImageOperation VOperation, typename TInputPixel1, typename TInputPixel2, typename TOutputPixel>
void
LaunchBinaryKernel(itk::SizeValueType   numberOfPixels,
                   const TInputPixel1 * input1,
                   const TInputPixel2 * input2,
                   TInputPixel1         constant1,
                   TInputPixel2         constant2,
                   TOutputPixel *       output,
                   TInputPixel2         divideAlmostZeroTolerance,
                   TInputPixel2         divideOrZeroOutThreshold,
                   TOutputPixel         divideOrZeroOutConstant,
                   TOutputPixel         divideByZeroValue)
{
  if (numberOfPixels == 0)
  {
    return;
  }

  constexpr unsigned int threadsPerBlock = 256;
  const auto             blocks = static_cast<unsigned int>((numberOfPixels + threadsPerBlock - 1) / threadsPerBlock);
  CudaBinaryImageFilterKernel<VOperation, TInputPixel1, TInputPixel2, TOutputPixel>
    <<<blocks, threadsPerBlock>>>(numberOfPixels,
                                  input1,
                                  input2,
                                  constant1,
                                  constant2,
                                  output,
                                  divideAlmostZeroTolerance,
                                  divideOrZeroOutThreshold,
                                  divideOrZeroOutConstant,
                                  divideByZeroValue);
  CUDA_CHECK(cudaGetLastError());
}

// Instantiate the supported operation/pixel-type combinations.
#define ITK_CUDA_BINARY_INSTANTIATE(OP, T)                         \
  template void CudaCommon_EXPORT LaunchBinaryKernel<OP, T, T, T>( \
    itk::SizeValueType, const T *, const T *, T, T, T *, T, T, T, T)

#define ITK_CUDA_BINARY_INSTANTIATE_ALL(T)                            \
  ITK_CUDA_BINARY_INSTANTIATE(CudaBinaryImageOperation::Add, T);      \
  ITK_CUDA_BINARY_INSTANTIATE(CudaBinaryImageOperation::Subtract, T); \
  ITK_CUDA_BINARY_INSTANTIATE(CudaBinaryImageOperation::Multiply, T); \
  ITK_CUDA_BINARY_INSTANTIATE(CudaBinaryImageOperation::Divide, T);   \
  ITK_CUDA_BINARY_INSTANTIATE(CudaBinaryImageOperation::DivideOrZeroOut, T)

ITK_CUDA_BINARY_INSTANTIATE_ALL(float);
ITK_CUDA_BINARY_INSTANTIATE_ALL(double);
ITK_CUDA_BINARY_INSTANTIATE_ALL(int);
ITK_CUDA_BINARY_INSTANTIATE_ALL(unsigned int);
ITK_CUDA_BINARY_INSTANTIATE_ALL(unsigned long);
ITK_CUDA_BINARY_INSTANTIATE_ALL(short);
ITK_CUDA_BINARY_INSTANTIATE_ALL(unsigned short);
ITK_CUDA_BINARY_INSTANTIATE_ALL(unsigned char);

#undef ITK_CUDA_BINARY_INSTANTIATE_ALL
#undef ITK_CUDA_BINARY_INSTANTIATE

} // namespace itk
