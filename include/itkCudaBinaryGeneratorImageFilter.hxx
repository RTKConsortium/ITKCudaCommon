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
#ifndef itkCudaBinaryGeneratorImageFilter_hxx
#define itkCudaBinaryGeneratorImageFilter_hxx

#include "itkNumericTraits.h"

#include <type_traits>

namespace itk
{

template <typename TInputImage1, typename TInputImage2, typename TOutputImage, typename TParentImageFilter>
template <CudaBinaryImageOperation VOperation>
void
CudaBinaryGeneratorImageFilter<TInputImage1, TInputImage2, TOutputImage, TParentImageFilter>::LaunchOperation()
{
  if constexpr (!std::is_same_v<TInputImage1, typename CudaTraits<TInputImage1>::Type> ||
                !std::is_same_v<TInputImage2, typename CudaTraits<TInputImage2>::Type> ||
                !std::is_same_v<TOutputImage, typename CudaTraits<TOutputImage>::Type>)
  {
    itkExceptionMacro("CUDA binary generator filters require itk::CudaImage inputs and output");
  }
  else
  {
    const auto * input1 = dynamic_cast<const TInputImage1 *>(this->ProcessObject::GetInput(0));
    const auto * input2 = dynamic_cast<const TInputImage2 *>(this->ProcessObject::GetInput(1));
    if (input1 == nullptr && input2 == nullptr)
    {
      itkExceptionMacro("At least one image input is required by a CUDA binary generator filter");
    }
    const auto   constant1 = input1 ? Input1ImagePixelType{} : this->GetConstant1();
    const auto   constant2 = input2 ? Input2ImagePixelType{} : this->GetConstant2();
    auto *       output = this->GetOutput();
    const auto & region = output->GetRequestedRegion();
    if ((input1 && input1->GetBufferedRegion() != region) || (input2 && input2->GetBufferedRegion() != region) ||
        output->GetBufferedRegion() != region)
    {
      itkExceptionMacro("CUDA arithmetic requires each buffered region to match the output requested region "
                        "(index and size)");
    }
    const auto numberOfPixels = region.GetNumberOfPixels();
    if (numberOfPixels == 0)
    {
      return;
    }
    const auto * input1Buffer =
      input1 ? static_cast<const Input1ImagePixelType *>(input1->GetCudaDataManager()->GetGPUBufferPointer()) : nullptr;
    const auto * input2Buffer =
      input2 ? static_cast<const Input2ImagePixelType *>(input2->GetCudaDataManager()->GetGPUBufferPointer()) : nullptr;
    auto * outputBuffer = static_cast<OutputImagePixelType *>(output->GetCudaDataManager()->GetGPUBufferPointer());

    Input2ImagePixelType divideAlmostZeroTolerance{};
    Input2ImagePixelType divideOrZeroOutThreshold{};
    OutputImagePixelType divideOrZeroOutConstant{};
    OutputImagePixelType divideByZeroValue{};

    if constexpr (VOperation == CudaBinaryImageOperation::Divide)
    {
      if constexpr (std::is_floating_point_v<Input2ImagePixelType>)
      {
        divideAlmostZeroTolerance =
          static_cast<Input2ImagePixelType>(0.1) * NumericTraits<Input2ImagePixelType>::epsilon();
      }
      divideByZeroValue = NumericTraits<OutputImagePixelType>::max();
    }
    else if constexpr (VOperation == CudaBinaryImageOperation::DivideOrZeroOut)
    {
      divideOrZeroOutThreshold = this->GetThreshold();
      divideOrZeroOutConstant = this->GetConstant();
    }

    LaunchBinaryKernel<VOperation, Input1ImagePixelType, Input2ImagePixelType, OutputImagePixelType>(
      numberOfPixels,
      input1Buffer,
      input2Buffer,
      constant1,
      constant2,
      outputBuffer,
      divideAlmostZeroTolerance,
      divideOrZeroOutThreshold,
      divideOrZeroOutConstant,
      divideByZeroValue);
  }
}

} // namespace itk

#endif
