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
#ifndef itkCudaSquareImageFilter_h
#define itkCudaSquareImageFilter_h

#include "itkCudaUnaryGeneratorImageFilter.h"
#include "itkSquareImageFilter.h"

namespace itk
{

/** \class CudaSquareImageFilter
 * \brief Cuda version of itk::SquareImageFilter.
 *
 * Computes the squared value of each pixel on the GPU. See
 * CudaUnaryGeneratorImageFilter.
 *
 * \sa CudaUnaryGeneratorImageFilter
 * \sa itk::SquareImageFilter
 *
 * \ingroup ITKCudaCommon
 */
template <typename TInputImage, typename TOutputImage = TInputImage>
class ITK_TEMPLATE_EXPORT CudaSquareImageFilter
  : public CudaUnaryGeneratorImageFilter<TInputImage, TOutputImage, SquareImageFilter<TInputImage, TOutputImage>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaSquareImageFilter);

  using Self = CudaSquareImageFilter;
  using Superclass =
    CudaUnaryGeneratorImageFilter<TInputImage, TOutputImage, SquareImageFilter<TInputImage, TOutputImage>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(CudaSquareImageFilter);

  using InputImagePixelType = typename Superclass::InputImagePixelType;
  using OutputImagePixelType = typename Superclass::OutputImagePixelType;

protected:
  CudaSquareImageFilter() = default;
  ~CudaSquareImageFilter() override = default;

  void
  GPUGenerateData() override
  {
    this->template LaunchOperation<CudaUnaryImageOperation::Square>();
  }
};

} // namespace itk

#endif
