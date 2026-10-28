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
#ifndef itkCudaUnaryGeneratorImageFilter_hxx
#define itkCudaUnaryGeneratorImageFilter_hxx

#include <type_traits>

namespace itk
{
template <typename TInputImage, typename TOutputImage, typename TParentImageFilter>
template <CudaUnaryImageOperation VOperation>
void
CudaUnaryGeneratorImageFilter<TInputImage, TOutputImage, TParentImageFilter>::LaunchOperation()
{
  if constexpr (!std::is_same_v<TInputImage, typename CudaTraits<TInputImage>::Type> ||
                !std::is_same_v<TOutputImage, typename CudaTraits<TOutputImage>::Type>)
  {
    itkExceptionMacro("CUDA unary generator filters require itk::CudaImage input and output");
  }
  else
  {
    const auto * input = this->GetInput();
    auto *       output = this->GetOutput();
    const auto & region = output->GetRequestedRegion();
    if (input->GetBufferedRegion() != region || output->GetBufferedRegion() != region)
    {
      itkExceptionMacro("CUDA arithmetic requires each buffered region to match the output requested region "
                        "(index and size)");
    }
    const auto numberOfPixels = region.GetNumberOfPixels();
    if (numberOfPixels == 0)
    {
      return;
    }
    const auto * inputBuffer =
      static_cast<const InputImagePixelType *>(input->GetCudaDataManager()->GetGPUBufferPointer());
    auto * outputBuffer = static_cast<OutputImagePixelType *>(output->GetCudaDataManager()->GetGPUBufferPointer());

    InputImagePixelType  lower{};
    InputImagePixelType  upper{};
    OutputImagePixelType outsideValue{};
    if constexpr (VOperation == CudaUnaryImageOperation::Threshold)
    {
      lower = this->GetLower();
      upper = this->GetUpper();
      outsideValue = this->GetOutsideValue();
    }
    LaunchUnaryKernel<VOperation, InputImagePixelType, OutputImagePixelType>(
      numberOfPixels, inputBuffer, outputBuffer, lower, upper, outsideValue);
  }
}

} // namespace itk

#endif
