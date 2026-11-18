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
#ifndef itkCudaDivideOrZeroOutImageFilter_h
#define itkCudaDivideOrZeroOutImageFilter_h

#include "itkCudaBinaryImageToImageFilter.h"
#include "itkCudaBinaryImageOperators.h"
#include "itkDivideOrZeroOutImageFilter.h"

#include <type_traits>

namespace itk
{

/** \class CudaDivideOrZeroOutImageFilter
 * \brief CUDA implementation of itk::DivideOrZeroOutImageFilter.
 *
 * Uses the inherited threshold and output constant. Supports two images or
 * an image and a constant, with the same pixel type. The default threshold
 * is 1 for integer pixel types, as in ITK 6.
 *
 * \ingroup ITKCudaCommon
 */
template <class TImage>
class ITK_TEMPLATE_EXPORT CudaDivideOrZeroOutImageFilter
  : public CudaBinaryImageToImageFilter<TImage, TImage, DivideOrZeroOutImageFilter<TImage, TImage, TImage>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaDivideOrZeroOutImageFilter);

  using Self = CudaDivideOrZeroOutImageFilter;
  using Superclass = CudaBinaryImageToImageFilter<TImage, TImage, DivideOrZeroOutImageFilter<TImage, TImage, TImage>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(CudaDivideOrZeroOutImageFilter);

protected:
  using PixelType = typename Superclass::PixelType;

  CudaDivideOrZeroOutImageFilter()
  {
    // Earlier ITK versions truncate the default threshold, 1e-5, to 0.
    if constexpr (std::is_integral_v<PixelType>)
    {
      this->SetThreshold(NumericTraits<PixelType>::OneValue());
    }
  }
  ~CudaDivideOrZeroOutImageFilter() override = default;

  void
  GenerateBinaryData(const PixelType *  input1,
                     const PixelType *  input2,
                     PixelType          constant1,
                     PixelType          constant2,
                     PixelType *        output,
                     itk::SizeValueType numberOfPixels) override
  {
    CudaDivideOrZeroOutImage(
      numberOfPixels, input1, input2, output, constant1, constant2, this->GetThreshold(), this->GetConstant());
  }
};

} // namespace itk

#endif
