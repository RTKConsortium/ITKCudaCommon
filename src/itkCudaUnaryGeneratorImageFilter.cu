/*=========================================================================*
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

#include "itkCudaUnaryGeneratorImageFilter.hcu"
#include "itkCudaUtil.h"

#include "itkNumericTraits.h"

#include "CudaCommonExport.h"

namespace itk
{
namespace
{

template <CudaUnaryImageOperation VOperation, typename TInputPixel, typename TOutputPixel>
__global__ void
CudaUnaryImageFilterKernel(itk::SizeValueType  numberOfPixels,
                           const TInputPixel * input,
                           TOutputPixel *      output,
                           TInputPixel         lower,
                           TInputPixel         upper,
                           TOutputPixel        outsideValue)
{
  const itk::SizeValueType index = static_cast<itk::SizeValueType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= numberOfPixels)
  {
    return;
  }
  if constexpr (VOperation == CudaUnaryImageOperation::Threshold)
  {
    const TInputPixel value = input[index];
    output[index] = lower <= value && value <= upper ? static_cast<TOutputPixel>(value) : outsideValue;
  }
  else if constexpr (VOperation == CudaUnaryImageOperation::Square)
  {
    // Squares in RealType and casts to the output, as itk::Functor::Square does.
    const auto realValue = static_cast<typename NumericTraits<TInputPixel>::RealType>(input[index]);
    output[index] = static_cast<TOutputPixel>(realValue * realValue);
  }
}

} // namespace

template <CudaUnaryImageOperation VOperation, typename TInputPixel, typename TOutputPixel>
void
LaunchUnaryKernel(itk::SizeValueType  numberOfPixels,
                  const TInputPixel * input,
                  TOutputPixel *      output,
                  TInputPixel         lower,
                  TInputPixel         upper,
                  TOutputPixel        outsideValue)
{
  if (numberOfPixels == 0)
  {
    return;
  }

  constexpr unsigned int threadsPerBlock = 256;
  const auto             blocks = static_cast<unsigned int>((numberOfPixels + threadsPerBlock - 1) / threadsPerBlock);
  CudaUnaryImageFilterKernel<VOperation, TInputPixel, TOutputPixel>
    <<<blocks, threadsPerBlock>>>(numberOfPixels, input, output, lower, upper, outsideValue);
  CUDA_CHECK(cudaGetLastError());
}

// Instantiate the supported operation/pixel-type combinations.
#define ITK_CUDA_UNARY_INSTANTIATE(OP, T) \
  template void CudaCommon_EXPORT LaunchUnaryKernel<OP, T, T>(itk::SizeValueType, const T *, T *, T, T, T)

#define ITK_CUDA_UNARY_INSTANTIATE_ALL(T)                            \
  ITK_CUDA_UNARY_INSTANTIATE(CudaUnaryImageOperation::Threshold, T); \
  ITK_CUDA_UNARY_INSTANTIATE(CudaUnaryImageOperation::Square, T)

ITK_CUDA_UNARY_INSTANTIATE_ALL(float);
ITK_CUDA_UNARY_INSTANTIATE_ALL(double);
ITK_CUDA_UNARY_INSTANTIATE_ALL(int);
ITK_CUDA_UNARY_INSTANTIATE_ALL(unsigned int);
ITK_CUDA_UNARY_INSTANTIATE_ALL(unsigned long);
ITK_CUDA_UNARY_INSTANTIATE_ALL(short);
ITK_CUDA_UNARY_INSTANTIATE_ALL(unsigned short);
ITK_CUDA_UNARY_INSTANTIATE_ALL(unsigned char);

#undef ITK_CUDA_UNARY_INSTANTIATE_ALL
#undef ITK_CUDA_UNARY_INSTANTIATE

} // namespace itk
