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
#ifndef itkCudaMultiplyImageFilter_h
#define itkCudaMultiplyImageFilter_h

#include "itkCudaBinaryGeneratorImageFilter.h"
#include "itkMultiplyImageFilter.h"

namespace itk
{

template <typename TInputImage1, typename TInputImage2 = TInputImage1, typename TOutputImage = TInputImage1>
class ITK_TEMPLATE_EXPORT CudaMultiplyImageFilter
  : public CudaBinaryGeneratorImageFilter<TInputImage1,
                                          TInputImage2,
                                          TOutputImage,
                                          MultiplyImageFilter<TInputImage1, TInputImage2, TOutputImage>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaMultiplyImageFilter);

  using Self = CudaMultiplyImageFilter;
  using Superclass = CudaBinaryGeneratorImageFilter<TInputImage1,
                                                    TInputImage2,
                                                    TOutputImage,
                                                    MultiplyImageFilter<TInputImage1, TInputImage2, TOutputImage>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(CudaMultiplyImageFilter);

  using Input1ImagePixelType = typename Superclass::Input1ImagePixelType;
  using Input2ImagePixelType = typename Superclass::Input2ImagePixelType;
  using OutputImagePixelType = typename Superclass::OutputImagePixelType;

protected:
  CudaMultiplyImageFilter() = default;
  ~CudaMultiplyImageFilter() override = default;

  void
  GPUGenerateData() override
  {
    this->template LaunchOperation<CudaBinaryImageOperation::Multiply>();
  }
};

} // namespace itk

#endif
