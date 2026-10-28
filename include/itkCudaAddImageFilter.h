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
#ifndef itkCudaAddImageFilter_h
#define itkCudaAddImageFilter_h

#include "itkAddImageFilter.h"
#include "itkCudaBinaryGeneratorImageFilter.h"

namespace itk
{

template <typename TInputImage1, typename TInputImage2 = TInputImage1, typename TOutputImage = TInputImage1>
class ITK_TEMPLATE_EXPORT CudaAddImageFilter
  : public CudaBinaryGeneratorImageFilter<TInputImage1,
                                          TInputImage2,
                                          TOutputImage,
                                          AddImageFilter<TInputImage1, TInputImage2, TOutputImage>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaAddImageFilter);

  using Self = CudaAddImageFilter;
  using Superclass = CudaBinaryGeneratorImageFilter<TInputImage1,
                                                    TInputImage2,
                                                    TOutputImage,
                                                    AddImageFilter<TInputImage1, TInputImage2, TOutputImage>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(CudaAddImageFilter);

  using Input1ImagePixelType = typename Superclass::Input1ImagePixelType;
  using Input2ImagePixelType = typename Superclass::Input2ImagePixelType;
  using OutputImagePixelType = typename Superclass::OutputImagePixelType;

protected:
  CudaAddImageFilter() = default;
  ~CudaAddImageFilter() override = default;

  void
  GPUGenerateData() override
  {
    this->template LaunchOperation<CudaBinaryImageOperation::Add>();
  }
};

} // namespace itk

#endif
